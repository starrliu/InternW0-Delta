#!/usr/bin/env python
"""Prepare one real CogACT window and test the full model at batch size one.

The prepare stage uses episode-local normalization solely for this feasibility
check. It is deliberately not a replacement for full-dataset training stats.
No distributed jobs, formal training schedule, or checkpoint publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torchvision.transforms import functional as TF

from wam.datasets.cogact_joint import compact_action, compact_state, sample_joint_window, scatter80
from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer

VIEWS = ("head_color", "hand_left_color", "hand_right_color")


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def normalizer(values):
    stats = {
        "mean": values.mean(0), "std": values.std(0, unbiased=False),
        "min": values.amin(0), "max": values.amax(0),
    }
    return SingleFieldLinearNormalizer(stats, "z-score"), stats


def read_views(lmdb_path, video_path, indices):
    import cv2
    import lmdb

    result = []
    with lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False) as env:
        with env.begin() as txn:
            for view in VIEWS:
                frames = []
                for index in indices:
                    key = f"{video_path}:{view}:{int(index):05d}".encode()
                    encoded = txn.get(key)
                    if encoded is None:
                        raise KeyError(f"Missing real image: {key.decode()}")
                    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Cannot decode {key.decode()}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if view == "head_color" and video_path.split("/", 1)[0] == "3173":
                        if image.shape[0] == 800:
                            image = image[40:760]
                        elif image.shape[0] == 500:
                            image = image[25:475]
                    frame = torch.from_numpy(image.copy()).permute(2, 0, 1)
                    # As in the ACONe recipe: resize before packing the canvas.
                    frames.append(TF.resize(frame.float() / 255.0, [192, 256], antialias=True))
                result.append(torch.stack(frames))
    return torch.stack(result)  # [view,time,C,H,W]


def canvas(views):
    """The existing robotwin pixel layout: head above the two wrist views."""
    head = TF.resize(views[0], [256, 256], antialias=True)
    left = TF.resize(views[1], [128, 128], antialias=True)
    right = TF.resize(views[2], [128, 128], antialias=True)
    image = torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2)
    return (image * 2 - 1).permute(1, 0, 2, 3).contiguous()


def prepare(args):
    metadata = np.load(args.metadata, allow_pickle=True).item()
    episode = args.episode
    quality = str(metadata.get("quality", ["good"] * len(metadata["video_path"]))[episode])
    if quality not in {"good", "medium"}:
        raise ValueError(f"Episode quality is {quality}")
    video_path = str(metadata["video_path"][episode])
    task = str(metadata["instructions"][episode]).strip().lower()
    start, end = map(int, metadata["start_end"][episode])
    current = start + args.offset
    action_dir = Path(args.action_dir)
    record_meta = json.loads((action_dir / "meta_data.json").read_text())[video_path]
    if sum(record_meta["dim_list"]) != 56:
        raise ValueError("Expected the 56-float CogACT record format")
    action_path = action_dir / video_path / "action.npy"
    if action_path.stat().st_size != record_meta["length"] * 56 * 4:
        raise ValueError("Action record length does not match metadata")
    records = np.memmap(action_path, mode="r", dtype=np.float32, shape=(record_meta["length"], 56))
    raw = sample_joint_window(records, start=start, end=end, current=current)
    state_norm, state_stats = normalizer(compact_state(records[start:end]))
    action_norm, action_stats = normalizer(compact_action(records[start:end-1], records[start+1:end]))
    action, action_mask = scatter80(action_norm.forward(raw["action"]))
    state, state_mask = scatter80(state_norm.forward(raw["state"]))

    recent_is_pad = current - 32 < start
    recent = max(start, current - 32)
    image_offsets = np.arange(0, 33, 4)
    image_indices = np.clip(current + image_offsets, start, end - 1)
    indices = np.concatenate([[start, recent], image_indices])
    views = read_views(Path(args.dataset_dir) / "lmdb/frames.lmdb", video_path, indices)
    memory_state, _ = scatter80(state_norm.forward(compact_state(records[[start, recent]])))
    prompt = DEFAULT_PROMPT.format(task=task)
    sample = {
        "sample_id": 0,
        "video": canvas(views[:, 2:]),
        "action": action,
        "proprio": state[:-1],
        "prompt": prompt,
        "video_layout": "single", "video_view_names": "canvas",
        "video_canvas_layout": "robotwin", "video_canvas_view_names": "|".join(VIEWS),
        "vlm_current_images": (views[:, 2] * 255).round().clamp(0, 255).to(torch.uint8),
        "image_is_pad": torch.from_numpy(current + image_offsets >= end),
        "action_is_pad": raw["action_is_pad"], "action_dim_is_pad": action_mask,
        "proprio_is_pad": raw["state_is_pad"][:-1], "proprio_dim_is_pad": state_mask,
        "memory_video_anchor": canvas(views[:, :1]),
        "memory_video_anchor_is_pad": torch.tensor([False]),
        "memory_video_anchor_proprio": memory_state[:1],
        "memory_video_anchor_proprio_is_pad": torch.tensor([False]),
        "memory_video_recent": canvas(views[:, 1:2]),
        "memory_video_recent_is_pad": torch.tensor([recent_is_pad]),
        "memory_video_recent_proprio": memory_state[1:2],
        "memory_video_recent_proprio_is_pad": torch.tensor([recent_is_pad]),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(sample, output / "sample.pt")
    write_json(output / "normalization.json", {
        "scope": "single episode; smoke only", "mode": "z-score",
        "state": {k: v.tolist() for k, v in state_stats.items()},
        "action": {k: v.tolist() for k, v in action_stats.items()},
    })
    report = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "metadata": str(Path(args.metadata).resolve()), "episode": episode,
        "video_path": video_path, "start": start, "end": end, "current": current,
        "task": task, "quality": quality, "action_file": str(action_path.resolve()),
        "action_file_sha256": hashlib.sha256(action_path.read_bytes()).hexdigest(),
        "command_vs_measured_joint_max_abs": float(np.abs(records[:, 2:16] - records[:, 38:52]).max()),
        "joint_semantics": "absolute next measured target; native source units, no unit conversion",
        "gripper_semantics": "current command; measured state divided by 120",
        "camera_extrinsics": False, "horizon": 32, "future_video_frames": 9,
        "normalization_scope": "single episode; smoke only", "terminal_repeat_valid_steps": 0,
        "shapes": {k: list(v.shape) for k, v in sample.items() if isinstance(v, torch.Tensor)},
    }
    write_json(output / "sample_report.json", report)
    # Save the actual current canvas for inspection.
    TF.to_pil_image((sample["video"][:, 0] + 1) / 2).save(output / "current.png")
    print(json.dumps(report, indent=2), flush=True)


def encode_text(args):
    report = json.loads((Path(args.sample_dir) / "sample_report.json").read_text())
    cache = str((Path(args.sample_dir) / "text").resolve())
    # Reuse the repository's text encoder/cache writer with one real instruction.
    subprocess.run([
        sys.executable, str(ROOT / "tools/text_cache.py"), "task=rtc",
        "model.redirect_common_files=false",
        "data.train.text_embedding_cache_dir=" + cache,
        "+override_instruction=" + json.dumps(report["task"]),
    ], cwd=ROOT, check=True)


def run(args):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import default_collate
    from wam.training.checkpoint import load_wam_checkpoint_strict
    from wam.training.module import WAMTrainingModule
    from wam.utils.logging_config import setup_logging

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "running", "stage": "preparation", "batch_size": 1}
    write_json(output / "result.json", report)
    cuda_ready = False

    def stage(name):
        report["stage"] = name
        report["elapsed_seconds"] = time.monotonic() - started
        if cuda_ready:
            torch.cuda.synchronize()
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        write_json(output / "result.json", report)
        print(f"[smoke] {name}", flush=True)

    try:
        setup_logging()
        cfg = OmegaConf.load(ROOT / "configs/smoke/cogact_5080.yaml")
        cfg.model = OmegaConf.merge(OmegaConf.load(ROOT / "configs/model/wam.yaml"), cfg.model)
        OmegaConf.resolve(cfg)
        OmegaConf.save(cfg, output / "config.yaml")
        sample_dir = Path(args.sample_dir)
        sample = torch.load(sample_dir / "sample.pt", weights_only=True, map_location="cpu")
        hashed = hashlib.sha256(sample["prompt"].encode()).hexdigest()
        context_path = sample_dir / "text" / f"{hashed}.t5_len128.wan22ti2v5b.pt"
        context = torch.load(context_path, weights_only=True, map_location="cpu")
        sample["context"] = context["context"].clone().masked_fill(~context["mask"].bool().unsqueeze(-1), 0)
        sample["context_mask"] = torch.ones_like(context["mask"], dtype=torch.bool)
        batch = default_collate([sample])
        device = torch.device("cuda:0")
        stage("cuda_initialization")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        cuda_ready = True
        report.update({
            "gpu": torch.cuda.get_device_name(device),
            "total_memory_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "horizon": 32, "precision": "bf16", "gradient_checkpointing": True,
            "full_vlm_enabled": True, "cpu_offload": False,
            "checkpoint": str(cfg.checkpoint), "normalization_scope": "single episode; smoke only",
        })
        stage("model_initialization")
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=str(device))
        stage("load_pretrained_checkpoint")
        report["checkpoint_coverage"] = load_wam_checkpoint_strict(model, cfg.checkpoint)
        model.set_trainable_modules()
        trainable = list(model.trainable_parameters())
        report["trainable_parameters"] = sum(p.numel() for p in trainable)
        report["trainable_weights_bf16_gib"] = sum(p.numel() * p.element_size() for p in trainable) / 2**30
        optimizer = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)
        module = WAMTrainingModule(model)
        before = torch.cat([p.detach().flatten()[:16].clone() for p in trainable])
        stage("forward")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = module(batch, global_step=0)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite loss")
        report["loss"] = float(loss.detach())
        report["metrics"] = {k: float(v) for k, v in metrics.items()}
        stage("backward")
        loss.backward()
        gradients = [p for p in trainable if p.grad is not None]
        if not gradients:
            raise RuntimeError("No trainable gradients")
        norm = torch.nn.utils.clip_grad_norm_(gradients, 1.0, error_if_nonfinite=True)
        report["gradient_norm"] = float(norm)
        stage("optimizer_step")
        optimizer.step()
        if not bool(torch.stack([torch.isfinite(p).all() for p in trainable]).all()):
            raise RuntimeError("Nonfinite updated parameters")
        after = torch.cat([p.detach().flatten()[:16] for p in trainable])
        if torch.equal(before, after):
            raise RuntimeError("No parameter update detected")
        report["status"] = "passed"
        stage("complete")
    except Exception as exc:
        report["status"] = "cuda_oom" if "out of memory" in str(exc).lower() else "error"
        report["error"] = str(exc)
        report["elapsed_seconds"] = time.monotonic() - started
        if cuda_ready:
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        write_json(output / "result.json", report)
        traceback.print_exc()
        return 1
    return 0


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--action-dir", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--offset", type=int, default=64)
    p.add_argument("--output", default=".cache/cogact_smoke")
    for name in ("encode-text", "run"):
        p = sub.add_parser(name)
        p.add_argument("--sample-dir", default=".cache/cogact_smoke")
        if name == "run":
            p.add_argument("--output", default="runs/cogact_5080_smoke")
    args = parser.parse_args()
    return {"prepare": prepare, "encode-text": encode_text, "run": run}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
