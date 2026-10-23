from pathlib import Path
import json
import runpy
from contextlib import nullcontext
import hashlib
from types import SimpleNamespace

import pytest
import torch

from wam.training.checkpoint import load_wam_checkpoint_strict

ROOT = Path(__file__).resolve().parents[1]
probe = runpy.run_path(str(ROOT / "tools/cogact_probe.py"))
smoke = runpy.run_path(str(ROOT / "tools/cogact_smoke.py"))


@pytest.mark.parametrize("maximum", [1, 3, 7, 16, 63])
def test_capacity_search_finds_integer_boundary(maximum):
    tested = []
    def attempt(batch):
        tested.append(batch)
        return "passed" if batch <= maximum else "cuda_oom"
    assert probe["search_capacity"](attempt, 64) == (maximum, maximum + 1)
    assert maximum in tested and maximum + 1 in tested


def test_capacity_limit_is_reported_as_lower_bound():
    assert probe["search_capacity"](lambda _: "passed", 64) == (64, None)


def test_non_oom_error_does_not_become_capacity_result():
    with pytest.raises(RuntimeError, match="non-OOM"):
        probe["search_capacity"](lambda batch: "passed" if batch == 1 else "error")


def policy():
    return SimpleNamespace(mot=torch.nn.Linear(2, 2), proprio_encoder=torch.nn.Linear(2, 3),
                           action_proprio_encoder=None, track_bridge=None, understanding=None)


def test_strict_policy_load_requires_all_active_modules(tmp_path):
    model = policy()
    path = tmp_path / "weights.pt"
    torch.save({"mot": model.mot.state_dict()}, path)
    with pytest.raises(ValueError, match="proprio_encoder"):
        load_wam_checkpoint_strict(model, path)


def test_strict_policy_preflight_does_not_partially_mutate_model(tmp_path):
    source, target = policy(), policy()
    original = target.mot.weight.detach().clone()
    state = {"mot": source.mot.state_dict(), "proprio_encoder": source.proprio_encoder.state_dict()}
    state["proprio_encoder"]["weight"] = torch.zeros(9, 2)
    path = tmp_path / "weights.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="Incompatible"):
        load_wam_checkpoint_strict(target, path)
    torch.testing.assert_close(original, target.mot.weight)


def test_strict_policy_load_accepts_complete_checkpoint(tmp_path):
    source, target = policy(), policy()
    path = tmp_path / "weights.pt"
    torch.save({"mot": source.mot.state_dict(), "proprio_encoder": source.proprio_encoder.state_dict()}, path)
    assert load_wam_checkpoint_strict(target, path) == {"mot": 2, "proprio_encoder": 2}
    torch.testing.assert_close(source.mot.weight, target.mot.weight)


def test_smoke_preparation_error_replaces_stale_success(tmp_path):
    result = tmp_path / "result.json"
    result.write_text('{"status":"passed"}')
    code = smoke["run"](SimpleNamespace(output=str(tmp_path), sample_dir=str(tmp_path / "missing")))
    assert code == 1
    report = json.loads(result.read_text())
    assert report["status"] == "error" and report["stage"] == "preparation"


def test_finite_parameter_check_rejects_nan():
    values = [torch.nn.Parameter(torch.tensor([1.0])), torch.nn.Parameter(torch.tensor([float("nan")]))]
    assert not probe["finite_parameters"](values)


def test_smoke_rejects_finite_loss_with_infinite_gradient(tmp_path, monkeypatch):
    from wam.training.module import WAMTrainingModule
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.zeros(()))
        def set_trainable_modules(self):
            pass
        def trainable_parameters(self):
            return self.parameters()
    model = Tiny()
    sample_dir = tmp_path / "sample"
    (sample_dir / "text").mkdir(parents=True)
    torch.save({"prompt": "test"}, sample_dir / "sample.pt")
    digest = hashlib.sha256(b"test").hexdigest()
    torch.save({"context": torch.zeros(128, 4), "mask": torch.ones(128, dtype=torch.bool)},
               sample_dir / "text" / f"{digest}.t5_len128.wan22ti2v5b.pt")
    for name in ("set_device", "reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "mock")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: SimpleNamespace(total_memory=16*2**30))
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("hydra.utils.instantiate", lambda *args, **kwargs: model)
    monkeypatch.setattr("wam.training.checkpoint.load_wam_checkpoint_strict", lambda *args: {})
    monkeypatch.setattr(WAMTrainingModule, "forward", lambda self, *args, **kwargs: (self.wam.value.sqrt(), {}))
    output = tmp_path / "result"
    assert smoke["run"](SimpleNamespace(output=str(output), sample_dir=str(sample_dir))) == 1
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "error" and report["stage"] == "backward"
    assert torch.isfinite(model.value)
