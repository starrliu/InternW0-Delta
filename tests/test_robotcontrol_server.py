import io
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from wam.datasets.cogact_joint import JOINT_INDICES, compact_state, scatter80
from wam.datasets.cogact_video import canvas
from wam.inference.cogact_policy import (
    CogACTNormalizer, CogACTPolicy, place_model, prepare_observation, read_run,
)
from wam.inference.robotcontrol_server import PolicyService, action_response, create_app, parse_payload


@pytest.fixture
def normalizer(tmp_path):
    stats = dict(mode="z-score", signature="test-data", state=dict(mean=list(np.arange(16.) / 10), std=[2.] * 16),
                 action=dict(mean=list(np.arange(16.) / 20), std=[.5] * 16))
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps(stats))
    return CogACTNormalizer(path)


@pytest.fixture
def images():
    return [np.full((192, 256, 3), x, dtype=np.uint8) for x in (30, 120, 230)]


@pytest.fixture
def payload():
    return dict(task_description="Pick up the cup.", has_left=True, has_right=True,
                state=dict(ROBOT_LEFT_JOINTS=[.1] * 7, ROBOT_RIGHT_JOINTS=[-.2] * 7,
                           ROBOT_LEFT_GRIPPER=[.25], ROBOT_RIGHT_GRIPPER=[.75]))


class FakePolicy:
    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.calls = []

    def predict(self, observation, instruction, memory):
        self.calls.append((observation, instruction, memory))
        return np.tile(np.arange(16, dtype=np.float32) / 20, (32, 1))


def multipart(payload, images):
    result = {"json": (io.BytesIO(json.dumps(payload).encode()), "data.json", "application/json")}
    for i, image in enumerate(images):
        stream = io.BytesIO()
        Image.fromarray(image).save(stream, "JPEG")
        stream.seek(0)
        result[f"image_{i}"] = (stream, f"{i}.jpg", "image/jpeg")
    return result


def test_training_state_image_and_action_parity(normalizer, images, payload):
    _, state = parse_payload(payload)
    records = np.zeros((1, 56), dtype=np.float32)
    records[0, 38:45], records[0, 45:52] = state[:7], state[8:15]
    records[0, 54:56] = state[[7, 15]] * 120
    np.testing.assert_array_equal(state, compact_state(records)[0].numpy())
    observation = prepare_observation(images, state, normalizer)
    expected = scatter80(normalizer.state.forward(compact_state(records)))[0][0]
    torch.testing.assert_close(observation.proprio, expected)
    views = torch.stack([torch.from_numpy(x).permute(2, 0, 1).float() / 255 for x in images])
    torch.testing.assert_close(observation.image, canvas(views[:, None]))
    np.testing.assert_array_equal(observation.vlm_images.numpy(), (views * 255).to(torch.uint8).numpy())
    targets = torch.arange(32 * 16).reshape(32, 16).float() / 1000
    encoded = scatter80(normalizer.action.forward(targets))[0]
    np.testing.assert_allclose(normalizer.decode_action(encoded[None]), targets, atol=1e-6)
    assert observation.proprio[[i for i in range(80) if i not in JOINT_INDICES]].count_nonzero() == 0


def test_history_exact_missing_and_reset(normalizer, images, payload):
    policy = FakePolicy(normalizer)
    service = PolicyService(policy)
    service.process(images, dict(payload, frame_index=0), now=10)
    assert policy.calls[-1][2]["memory_video_recent_is_pad"].item()
    service.process(images, dict(payload, frame_index=16), now=11)
    service.process(images, dict(payload, frame_index=32), now=12)
    assert not policy.calls[-1][2]["memory_video_recent_is_pad"].item()
    service.process(images, dict(payload, frame_index=65), now=13)
    assert policy.calls[-1][2]["memory_video_recent_is_pad"].item()  # no frame 33
    with pytest.raises(ValueError, match="strictly increase"):
        service.process(images, dict(payload, frame_index=64), now=14)
    service.process(images, dict(payload, reset=True, frame_index=0), now=15)
    assert service.history.last_step == 0
    service.process(images, dict(payload, task_description="New task", frame_index=0), now=16)
    assert service.history.last_step == 0
    with pytest.raises(ValueError, match="mix"):
        service.process(images, dict(payload, task_description="New task"), now=17)


def test_stock_clock_uses_elapsed_time_not_assumed_chunk(normalizer, images, payload):
    service = PolicyService(FakePolicy(normalizer))
    service.process(images, payload, now=10)
    _, step, recent = service.process(images, payload, now=11.1)
    assert step == 33 and recent == 0
    _, step, recent = service.process(images, payload, now=15)
    assert step == 150 and recent is None
    _, step, recent = service.process(images, payload, now=80)
    assert step == 0 and recent is None


def test_no_rtc_and_model_input_contract(normalizer, images, payload, monkeypatch):
    seen = {}
    class Model:
        def encode_prompt(self, prompt):
            seen["encoded_prompt"] = prompt
            return torch.zeros(1, 128, 4096), torch.ones(1, 128, dtype=torch.bool)

    def infer(model, **kwargs):
        seen.update(kwargs)
        return {"action": torch.zeros(1, 32, 80)}

    monkeypatch.setattr("wam.inference.online_action_policy.infer_online_action_chunk", infer)
    service = PolicyService(CogACTPolicy(Model(), normalizer))
    service.process(images, payload, now=1)
    assert seen["input_image"].shape == (1, 3, 384, 256)
    assert seen["vlm_current_images"].shape == (3, 3, 192, 256)
    assert seen["proprio"].shape == (1, 80)
    assert seen["action_dim_is_pad"].shape == (80,)
    assert seen["action_horizon"] == 32 and seen["num_inference_steps"] == 10
    assert "rtc_guidance" not in seen and "rtc_prefix_condition" not in seen
    assert "Pick up the cup." in seen["understanding_prompt"]


def test_separate_text_device_preserves_prompt_encoding(normalizer, images, payload, monkeypatch):
    from wam.model.world_action_model import WAM
    contexts = []
    class TextEncoder(torch.nn.Module):
        def forward(self, ids, mask):
            return ids[..., None].float().expand(-1, -1, 4096)
    class Model:
        device = torch.device("cpu")
        text_encoder = TextEncoder()
        tokenizer = staticmethod(lambda *args, **kwargs: (torch.tensor([[2, 3, 99]]), torch.tensor([[1, 1, 0]])))
        encode_prompt = WAM.encode_prompt

    def infer(model, **kwargs):
        contexts.append((kwargs["context"].clone(), kwargs["context_mask"].clone()))
        return {"action": torch.zeros(1, 32, 80)}
    monkeypatch.setattr("wam.inference.online_action_policy.infer_online_action_chunk", infer)
    for device in (None, "cpu"):
        PolicyService(CogACTPolicy(Model(), normalizer, text_device=device)).process(images, payload, now=1)
    for standard, separate in zip(contexts[0], contexts[1]):
        torch.testing.assert_close(standard, separate, rtol=0, atol=0)
    assert contexts[1][0][:, -1].count_nonzero() == 0
    assert contexts[1][1].all()


def test_device_placement_updates_plain_vae_and_vlm_attributes():
    model = torch.nn.Module()
    model.torch_dtype = torch.bfloat16
    model.mot = torch.nn.Linear(2, 2)
    model.text_encoder = torch.nn.Linear(2, 2)
    model.vae = torch.nn.Linear(2, 2)
    model.vae.mean = torch.tensor([1., 2.])
    model.vae.std = torch.tensor([2., 4.])
    model.understanding = torch.nn.Module()
    model.understanding.vlm = torch.nn.Linear(2, 2)
    place_model(model, "cpu", "cpu", "cpu")
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert model.vae.mean.dtype == model.vae.std.dtype == torch.bfloat16
    assert model.vae.scale[0] is model.vae.mean
    torch.testing.assert_close(model.vae.scale[1], 1 / model.vae.std)
    assert model.understanding.vlm.device == model.device == torch.device("cpu")
    assert model.understanding.vlm.dtype == model.torch_dtype


def test_http_contract_validation_reset_and_busy(normalizer, images, payload):
    policy = FakePolicy(normalizer)
    service = PolicyService(policy, report={"step": 80000})
    client = create_app(service).test_client()
    response = client.post("/api/inference", data=multipart(payload, images))
    assert response.status_code == 200
    assert {k: np.shape(v) for k, v in response.json.items()} == {
        "ROBOT_LEFT_JOINTS": (32, 7), "ROBOT_RIGHT_JOINTS": (32, 7),
        "ROBOT_LEFT_GRIPPER": (32, 1), "ROBOT_RIGHT_GRIPPER": (32, 1)}
    assert response.headers["X-WAM-Recent-Frame"] == "masked"
    assert client.get("/health").json["step"] == 80000
    assert client.post("/api/reset").status_code == 200
    assert service.history.anchor is None
    missing_image = multipart(payload, images)
    del missing_image["image_2"]
    assert client.post("/api/inference", data=missing_image).status_code == 400
    assert service.history.anchor is None
    assert client.post("/api/inference", data={"json": "{bad"}).status_code == 400
    with service.lock:
        assert client.post("/api/inference", data=multipart(payload, images)).status_code == 409
        assert client.post("/api/reset").status_code == 409
    response = client.post("/api/observe", data=multipart(dict(payload, frame_index=0), images))
    assert response.status_code == 200
    assert len(policy.calls) == 1


@pytest.mark.parametrize("value", [[0.] * 6, [float("nan")] * 7, [[0.]] * 7])
def test_reject_invalid_joints(payload, value):
    payload["state"]["ROBOT_LEFT_JOINTS"] = value
    with pytest.raises(ValueError):
        parse_payload(payload)


def test_reject_pose_only_gripper_units_and_flags(payload):
    for field, value in (("ROBOT_LEFT_GRIPPER", [120]), ("ROBOT_RIGHT_GRIPPER", [-.1])):
        invalid = dict(payload, state=dict(payload["state"], **{field: value}))
        with pytest.raises(ValueError, match=r"\[0,1\]"):
            parse_payload(invalid)
    with pytest.raises(ValueError, match="must be boolean"):
        parse_payload(dict(payload, has_left="false"))
    with pytest.raises(ValueError):
        parse_payload(dict(payload, state={"ROBOT_LEFT_TRANS": [0, 0, 0]}))


def test_inactive_arm_holds_state_and_nonfinite_outputs_rejected(payload):
    _, state = parse_payload(payload)
    result = action_response(np.ones((32, 16)) * 2, state, dict(payload, has_left=False))
    np.testing.assert_allclose(result["ROBOT_LEFT_JOINTS"], np.tile(state[:7], (32, 1)))
    assert result["ROBOT_LEFT_GRIPPER"][0] == [.25]
    assert result["ROBOT_RIGHT_GRIPPER"][0] == [1.]
    with pytest.raises(RuntimeError):
        action_response(np.full((32, 16), np.nan), state, payload)


def test_model_error_returns_500_and_clears_history(normalizer, images, payload):
    class BrokenPolicy(FakePolicy):
        def predict(self, *args):
            raise ValueError("bad model tensor")
    service = PolicyService(BrokenPolicy(normalizer))
    response = create_app(service).test_client().post("/api/inference", data=multipart(payload, images))
    assert response.status_code == 500
    assert "ROBOT_LEFT_JOINTS" not in response.json
    assert service.history.anchor is None


def test_read_current_training_config_and_manifest(tmp_path, normalizer):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        cfg = compose(config_name="train", overrides=["task=cogact_bs256_100k_2node"])
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    folder = tmp_path / "checkpoints/weights"
    folder.mkdir(parents=True)
    torch.save({"step": 80000, "mot": {}}, folder / "slot_0.pt")
    manifest = {"latest": {"path": "slot_0.pt", "step": 80000}}
    (folder / "weights_manifest.json").write_text(json.dumps(manifest))
    args = SimpleNamespace(run_dir=tmp_path, checkpoint="latest", stats=tmp_path / "normalization.json",
                           model_root=None, vlm_path=None)
    prepared = read_run(args)
    assert prepared[-1]["step"] == 80000 and prepared[-1]["rtc"] is False
    manifest["latest"]["step"] = 75000
    (folder / "weights_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completion manifest"):
        read_run(args)


def test_upstream_robotcontrol_client_and_interpreter(normalizer, images, payload):
    """Run with ROBOTCONTROL_G1_ROOT pointing to an unmodified checkout; no SDK/GPU."""
    root = os.environ.get("ROBOTCONTROL_G1_ROOT")
    if not root:
        pytest.skip("Set ROBOTCONTROL_G1_ROOT for the upstream HTTP compatibility test")
    sys.path.insert(0, str(Path(root).resolve()))
    from g1_platform.policy.cogact import CogACTPolicy as Client
    from g1_platform.core.types import ActionFormat, InputFormat
    from g1_platform.action.interpreter import ActionInterpreter
    from werkzeug.serving import make_server
    server = make_server("127.0.0.1", 0, create_app(PolicyService(FakePolicy(normalizer))))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = Client("127.0.0.1", server.server_port,
                        action_format=ActionFormat.JOINT_ANGLES, input_format=InputFormat.JOINT_ANGLES)
        result = client.step([Image.fromarray(x) for x in images], payload["task_description"], payload["state"])
        interpreter = ActionInterpreter(None, SimpleNamespace(head_extrinsic=np.eye(4)))
        interpreted = interpreter.interpret(result, client.action_format,
                                            SimpleNamespace(left_arm_joints=np.zeros(7), right_arm_joints=np.zeros(7)))
        assert len(interpreted.left_joints) == len(interpreted.right_gripper) == 32
        np.testing.assert_allclose(interpreted.right_joints[0], np.arange(8, 15) / 20)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        sys.path.remove(str(Path(root).resolve()))
