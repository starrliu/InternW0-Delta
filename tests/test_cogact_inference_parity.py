"""CPU checks for the conditions shared by CogACT training and inference.

These compare the actual attention builders and VLM conditioning paths. They
do not load a checkpoint or establish full-model GPU numerical equivalence.
"""

from types import SimpleNamespace

import pytest
import torch

from wam.datasets.cogact_video import VIEWS
from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wam.inference.online_action_policy import _build_mot_attention_mask
from wam.inference.qwen_vl_cache import _get_online_vlm_pack
from wam.model.modules.conditioning.frame_attention import (
    build_frame_memory_attention_mask,
)
from wam.model.modules.memory.proprio_encoder import ProprioContextEncoder
from wam.model.modules.understanding.sequence_conditioning import build_frame_vlm_pack


@pytest.mark.parametrize("tokens_per_frame", [3, 96])
@pytest.mark.parametrize("future_frames", [1, 2])
@pytest.mark.parametrize("with_delta", [False, True])
def test_online_attention_equals_training_without_future_teacher(
    tokens_per_frame, future_frames, with_delta
):
    # The actual CogACT recipe has 96 tokens/frame, three clean condition
    # frames, two future teacher latent frames, and 96 future-delta queries.
    condition_tokens = 3 * tokens_per_frame
    teacher_tokens = future_frames * tokens_per_frame
    delta_tokens = tokens_per_frame if with_delta else 0
    real_video_tokens = condition_tokens + teacher_tokens
    action_tokens = 32
    train_mask = build_frame_memory_attention_mask(
        video_seq_len=real_video_tokens + delta_tokens,
        action_seq_len=action_tokens,
        video_tokens_per_frame=tokens_per_frame,
        num_condition_frames=3,
        num_future_delta_tokens=delta_tokens,
        num_anchor_condition_frames=1,
        device=torch.device("cpu"),
    )
    online_mask = _build_mot_attention_mask(
        video_seq_len=condition_tokens + delta_tokens,
        action_seq_len=action_tokens,
        video_tokens_per_frame=tokens_per_frame,
        num_future_delta_tokens=delta_tokens,
        device=torch.device("cpu"),
    )
    keep = torch.cat(
        [
            torch.arange(condition_tokens),
            torch.arange(real_video_tokens, real_video_tokens + delta_tokens + action_tokens),
        ]
    )

    # Deleting the privileged teacher must preserve every visible edge and
    # must not remove a dependency of a condition, delta, or action query.
    torch.testing.assert_close(train_mask[keep][:, keep], online_mask, rtol=0, atol=0)
    assert not train_mask[keep, condition_tokens:real_video_tokens].any()
    assert train_mask[condition_tokens:real_video_tokens, condition_tokens:real_video_tokens].any()


class RecordingUnderstanding:
    """Replace only the expensive frozen backbone, recording its real inputs."""

    def __init__(self, dtype):
        self.calls = []
        self.context = torch.arange(32, dtype=dtype).reshape(1, 4, 8)
        self.mask = torch.tensor([[True, True, False, False]])

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"vlm_context": self.context.clone(), "vlm_mask": self.mask.clone()}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_middle_view", [False, True])
def test_training_and_online_vlm_conditions_match(dtype, mask_middle_view):
    understanding = RecordingUnderstanding(dtype)
    state_encoder = ProprioContextEncoder(80, 8).to(dtype=dtype)
    with torch.no_grad():
        state_encoder.weight.zero_()
        state_encoder.weight[:, :8].copy_(torch.eye(8, dtype=dtype))
        state_encoder.bias.fill_(0.25)
        state_encoder.type_embedding.fill_(0.5)
    model = SimpleNamespace(
        understanding_enabled=True,
        understanding_cfg={},
        understanding=understanding,
        action_proprio_encoder=state_encoder,
        proprio_dim=80,
        device=torch.device("cpu"),
        torch_dtype=dtype,
    )
    images = (torch.arange(3 * 3 * 192 * 256) % 256).to(torch.uint8).reshape(3, 3, 192, 256)
    state = torch.linspace(-1, 1, 80, dtype=dtype).reshape(1, 80)
    prompt = DEFAULT_PROMPT.format(task="Pick up the cup.")
    view_names = "|".join(VIEWS)
    view_pad = torch.tensor([False, True, False]) if mask_middle_view else None
    sample = {
        "vlm_current_images": images.unsqueeze(0),
        "current_proprio": state,
        "prompt": [prompt],
        "video_canvas_view_names": [view_names],
        "image_is_pad": torch.zeros(1, 9, dtype=torch.bool),
    }
    if view_pad is not None:
        sample["vlm_current_view_is_pad"] = view_pad.unsqueeze(0)

    training = build_frame_vlm_pack(model, inputs=sample)
    online = _get_online_vlm_pack(
        model,
        vlm_current_images=images,
        proprio=state,
        prompt=prompt,
        vlm_view_names=view_names,
        vlm_current_view_is_pad=view_pad,
    )

    assert len(understanding.calls) == 2
    train_call, online_call = understanding.calls
    for key in ("frames", "prompts", "view_names", "frame_labels", "valid_mask", "view_valid_mask"):
        left, right = train_call.get(key), online_call.get(key)
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert left == right, key
    for key in ("vlm_context", "vlm_mask"):
        torch.testing.assert_close(training[key], online[key], rtol=0, atol=0)

    # Confirm both paths appended the encoded current state, preserved the
    # backbone's padding mask, and marked the new state token as valid.
    torch.testing.assert_close(online["vlm_context"][:, :-1], understanding.context, rtol=0, atol=0)
    torch.testing.assert_close(online["vlm_context"][:, -1:], state_encoder(state.unsqueeze(1)), rtol=0, atol=0)
    assert online["vlm_mask"].tolist() == [[True, True, False, False, True]]
    assert online["vlm_mask_all_valid"] is False
