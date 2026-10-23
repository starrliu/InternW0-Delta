import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "cogact_prepare_dataset", Path(__file__).parents[1] / "tools/cogact_prepare_dataset.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_streaming_moments_matches_population_statistics():
    values = np.random.default_rng(91).normal(size=(101, 16)) * 0.03 + 10000
    moments = prepare.Moments()
    for part in np.array_split(values, 7):
        moments.add(part)
    result = moments.result()
    np.testing.assert_allclose(result["mean"], values.mean(0))
    np.testing.assert_allclose(result["std"], values.std(0), rtol=1e-9)
    assert result["count"] == len(values)
    with pytest.raises(ValueError, match="Nonfinite|nonfinite"):
        moments.add(np.full((1, 16), np.nan))


def test_preparation_uses_train_only_next_state_and_never_writes_source(tmp_path):
    source = tmp_path / "source"
    leaf = source / "leaf"
    actions = leaf / "actions_cmd_rfix_v1"
    actions.mkdir(parents=True)
    videos = {}
    for i in range(1000):
        key = f"task/recording_{i}"
        videos.setdefault(prepare.recording_split(key, .5, 42), key)
        if len(videos) == 2:
            break
    metadata = {"video_path": [], "instructions": [], "start_end": [], "quality": []}
    action_meta = {}
    for split, video in sorted(videos.items()):
        rows = np.zeros((5, 56), np.float32)
        rows[:, 38:52] = np.arange(5)[:, None] + (100 if split == "validation" else 0)
        rows[:, 20] = np.arange(5) * .1
        rows[:, 21] = np.arange(5) * .2
        rows[:, 54:56] = 60
        p = actions / video / "action.npy"
        p.parent.mkdir(parents=True)
        rows.tofile(p)
        action_meta[video] = {"length": 5, "dim_list": [56]}
        metadata["video_path"].append(video)
        metadata["instructions"].append("Pick up cup")
        metadata["start_end"].append([1, 5])
        metadata["quality"].append("good")
    np.save(leaf / "episodes.npy", metadata)
    (actions / "meta_data.json").write_text(json.dumps(action_meta))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset_root": str(source), "datasets": [
        {"dataset_folder": "leaf", "metadata_file": "episodes.npy"}]}))
    before = {str(p): prepare.sha256(p) for p in source.rglob("*") if p.is_file()}
    out = tmp_path / "prepared"
    report = prepare.prepare(manifest, out, .5, require_images=False)
    stats = json.loads((out / "normalization.json").read_text())
    assert report["counts"] == {"train": 3, "validation": 3}
    assert stats["state"]["count"] == 4
    assert stats["action"]["count"] == 3
    assert stats["state"]["mean"][0] == 2.5
    assert stats["action"]["mean"][0] == 3
    np.testing.assert_allclose(stats["action"]["mean"][7], .2)
    assert stats["state"]["mean"][7] == .5
    assert before == {str(p): prepare.sha256(p) for p in source.rglob("*") if p.is_file()}
    assert (out / "complete.json").is_file()
    with pytest.raises(ValueError, match="read-only"):
        prepare.prepare(manifest, source / "stats", .5, require_images=False)
    with pytest.raises(ValueError, match="immutable"):
        prepare.prepare(manifest, out, .5, require_images=False)
    all_out = tmp_path / "all_training"
    all_report = prepare.prepare(manifest, all_out, require_images=False)
    assert all_report["counts"] == {"train": 6}
    all_stats = json.loads((all_out / "normalization.json").read_text())
    assert all_stats["state"]["count"] == 8
    assert all_stats["state"]["mean"][0] == 52.5
    assert json.loads((all_out / "meta/tasks.jsonl").read_text())["task"] == "pick up cup"
    assert before == {str(p): prepare.sha256(p) for p in source.rglob("*") if p.is_file()}


def test_download_rejects_destination_escape(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "stage", Path(__file__).parents[1] / "tools/cogact_stage_weights.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    for relative in ("../source.bin", "/data/liluo/msra_process/a", "."):
        with pytest.raises(ValueError):
            stage.beneath(tmp_path, relative)
    assert stage.beneath(tmp_path, "base/pretrain.pt") == tmp_path / "base/pretrain.pt"
