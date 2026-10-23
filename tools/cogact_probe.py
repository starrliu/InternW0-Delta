#!/usr/bin/env python
"""Bounded eight-GPU CogACT training, capacity search, and loss-trend probe.

Uses real prepared windows and the native WAM loss, with DeepSpeed ZeRO-2.
Capacity applies to this fixed input geometry, precision and optimizer only.
The loss experiment fits one recording; it does not estimate generalization.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import default_collate


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args):
    from cogact_smoke import prepare as prepare_window

    source = Path(args.sample_dir)
    report = json.loads((source / "sample_report.json").read_text())
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    offsets = np.linspace(32, report["end"] - report["start"] - 33,
                          args.windows, dtype=int)
    if len(set(offsets.tolist())) != args.windows or min(offsets) < 32:
        raise ValueError("Not enough distinct complete windows in the source segment")
    entries = []
    for index, offset in enumerate(offsets):
        folder = destination / f"window_{index:03d}"
        current = report["start"] + int(offset)
        if not (folder / "sample_report.json").exists():
            with redirect_stdout(io.StringIO()):
                prepare_window(SimpleNamespace(
                    metadata=report["metadata"], episode=report["episode"],
                    offset=int(offset), action_dir=str(Path(report["action_file"]).parents[2]),
                    dataset_dir=report["dataset_dir"], output=str(folder)))
        prepared = json.loads((folder / "sample_report.json").read_text())
        if (prepared["current"] != current or prepared["action_file_sha256"] != report["action_file_sha256"]):
            raise ValueError(f"Existing window has incompatible source: {folder}")
        sample = torch.load(folder / "sample.pt", weights_only=True, mmap=True)
        if any(not bool(torch.isfinite(v).all()) for v in sample.values()
               if torch.is_tensor(v) and v.is_floating_point()):
            raise ValueError(f"Nonfinite source window {index}")
        if bool(sample["action_is_pad"].any()) or bool(sample["image_is_pad"].any()):
            raise ValueError("Capacity measurements require full unpadded windows")
        entries.append({"path": f"window_{index:03d}/sample.pt", "current": current,
                        "sha256": sha256(folder / "sample.pt")})
        print(f"[prepare] {index + 1}/{args.windows}", flush=True)
    shutil.copytree(source / "text", destination / "text", dirs_exist_ok=True)
    manifest = {"version": 1, "scope": "single-segment short-run fitting, not generalization",
                "source": report, "windows": entries,
                "geometry": {"horizon": 32, "video_frames": 9, "height": 384, "width": 256}}
    # Publish only after every input file is closed and hashed.
    write_json(destination / "manifest.json", manifest)


class Windows:
    def __init__(self, directory):
        self.root = Path(directory)
        self.digest = sha256(self.root / "manifest.json")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.entries = self.manifest["windows"]
        self.contexts = {}

    def sample(self, index):
        sample = torch.load(self.root / self.entries[index % len(self.entries)]["path"],
                            map_location="cpu", weights_only=True, mmap=True)
        digest = hashlib.sha256(sample["prompt"].encode()).hexdigest()
        if digest not in self.contexts:
            cache = torch.load(self.root / "text" / f"{digest}.t5_len128.wan22ti2v5b.pt",
                               map_location="cpu", weights_only=True)
            self.contexts[digest] = (
                cache["context"].clone().masked_fill(~cache["mask"].bool().unsqueeze(-1), 0),
                torch.ones_like(cache["mask"], dtype=torch.bool))
        sample["context"], sample["context_mask"] = self.contexts[digest]
        return sample

    def batch(self, step, rank, batch_size):
        # Deterministic sampling makes a resumed update see the same examples.
        order = torch.randperm(len(self.entries), generator=torch.Generator().manual_seed(42 + step))
        return default_collate([self.sample(int(order[(rank * batch_size + i) % len(order)]))
                                for i in range(batch_size)])


def parameter_sample(parameters):
    return torch.cat([p.detach().flatten()[:16].clone() for p in parameters])


def finite_parameters(parameters):
    return bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all())


@contextmanager
def fixed_noise(seed, device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[device]):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def worker(args):
    import deepspeed
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from wam.training.checkpoint import load_wam_checkpoint_strict
    from wam.training.module import WAMTrainingModule

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    output = Path(args.output)
    result_path = output / f"rank_{rank:02d}.json"
    report = {"status": "running", "stage": "preparation", "rank": rank,
              "micro_batch": args.batch_size, "global_batch": world * args.batch_size}
    write_json(result_path, report)
    device = torch.device("cuda", local_rank)
    started = time.monotonic()
    ready = False

    def stage(name):
        report["stage"] = name
        report["elapsed_seconds"] = time.monotonic() - started
        if ready:
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            report["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
        write_json(result_path, report)
        print(f"[rank {rank}] {name}", flush=True)

    try:
        if world != 8:
            raise ValueError(f"Expected 8 ranks, got {world}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        ready = True
        report.update(gpu=torch.cuda.get_device_name(device),
                      total_memory_gib=torch.cuda.get_device_properties(device).total_memory / 2**30)
        if "A100" not in report["gpu"]:
            raise ValueError(f"Expected A100, got {report['gpu']}")
        deepspeed.init_distributed(dist_backend="nccl", timeout=timedelta(minutes=5))
        torch.manual_seed(42)
        cfg = OmegaConf.load(ROOT / "configs/smoke/cogact_5080.yaml")
        cfg.model = OmegaConf.merge(OmegaConf.load(ROOT / "configs/model/wam.yaml"), cfg.model)
        OmegaConf.resolve(cfg)
        data = Windows(args.data)
        signature = {"model": OmegaConf.to_container(cfg.model), "data_sha256": data.digest,
                     "world_size": world, "micro_batch": args.batch_size, "lr": args.lr,
                     "zero_stage": 2, "precision": "bf16", "grad_accum": 1}
        signature_hash = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        if rank == 0:
            write_json(output / "configuration.json", signature)
        stage("model_initialization")
        core = instantiate(cfg.model, model_dtype=torch.bfloat16, device=str(device))
        stage("strict_pretrained_load")
        report["checkpoint_coverage"] = load_wam_checkpoint_strict(core, cfg.checkpoint)
        core.set_trainable_modules()
        parameters = list(core.trainable_parameters())
        report["trainable_parameters"] = sum(p.numel() for p in parameters)
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "train_batch_size": world * args.batch_size, "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True}, "gradient_clipping": 1.0,
            "zero_optimization": {"stage": 2, "overlap_comm": False,
                                  "contiguous_gradients": False,
                                  "reduce_bucket_size": 200000000,
                                  "allgather_bucket_size": 200000000},
            "steps_per_print": 1000000, "wall_clock_breakdown": False,
        }
        stage("deepspeed_initialization")
        engine, optimizer, _, scheduler = deepspeed.initialize(
            model=WAMTrainingModule(core), model_parameters=parameters,
            optimizer=optimizer, lr_scheduler=scheduler, config=ds_config)
        core.set_trainable_modules()
        checkpoint_root = output / "checkpoints"
        completed = sorted(checkpoint_root.glob("step_*/complete.json"))
        step = 0
        if args.mode == "train" and completed:
            checkpoint = completed[-1].parent
            marker = json.loads((checkpoint / "complete.json").read_text())
            if marker["signature"] != signature_hash:
                raise ValueError("Resume configuration or data do not match")
            for filename, size in marker["files"].items():
                if (checkpoint / filename).stat().st_size != size:
                    raise ValueError(f"Incomplete checkpoint file: {filename}")
            stage("resume")
            # Frozen backbones are excluded from training states and have just
            # been loaded from their original, checked policy/backbone assets.
            loaded, client = engine.load_checkpoint(str(checkpoint_root), tag=checkpoint.name,
                                                     load_module_strict=False)
            if loaded is None or client["signature"] != signature_hash:
                raise ValueError("Training state failed to restore")
            step = int(client["step"])
            if int(engine.global_steps) != step:
                raise ValueError("DeepSpeed and client step counters disagree on resume")
            rng = torch.load(checkpoint / f"rng_rank_{rank:02d}.pt", weights_only=False, map_location="cpu")
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state(rng["cuda"], device)
            report["resumed_from_step"] = step
        else:
            random.seed(42 + rank)
            np.random.seed(42 + rank)
            torch.manual_seed(42 + rank)
        initial = parameter_sample(parameters)
        step_times = []

        def average_metrics(loss, metrics):
            names = ["loss_total", *sorted(metrics)]
            values = torch.stack([loss.detach().float(), *[metrics[k].detach().float() for k in sorted(metrics)]])
            dist.all_reduce(values)
            return dict(zip(names, (values / world).cpu().tolist()))

        def evaluate(step):
            engine.eval()
            averages = []
            with torch.no_grad():
                for seed in (901, 902, 903):
                    with fixed_noise(seed + rank, local_rank), torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, metrics = engine(default_collate([data.sample(rank)]), global_step=step)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite fixed-noise evaluation loss")
                    averages.append(average_metrics(loss, metrics))
            engine.train()
            core.set_trainable_modules()
            result = {key: sum(row[key] for row in averages) / len(averages) for key in averages[0]}
            result.update(step=step, scope="same eight training windows and fixed noise; fitting only")
            if rank == 0:
                write_json(output / "eval" / f"step_{step:06d}.json", result)
                print("[fixed-eval] " + json.dumps(result), flush=True)

        def save(step):
            tag = f"step_{step:06d}"
            stage("checkpoint_save")
            engine.save_checkpoint(str(checkpoint_root), tag=tag,
                client_state={"step": step, "signature": signature_hash},
                save_latest=False, exclude_frozen_parameters=True)
            folder = checkpoint_root / tag
            torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                        "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)},
                       folder / f"rng_rank_{rank:02d}.pt")
            dist.barrier()
            if rank == 0:
                sizes = {str(p.relative_to(folder)): p.stat().st_size for p in folder.rglob("*") if p.is_file()}
                if not sizes or any(size == 0 for size in sizes.values()):
                    raise RuntimeError("Empty checkpoint output")
                write_json(folder / "complete.json", {"step": step, "signature": signature_hash, "files": sizes})
            dist.barrier()

        if args.mode == "train" and step == 0:
            evaluate(0)
        stop = min(args.steps, args.stop_at or args.steps)
        while step < stop:
            stage(f"train_step_{step + 1}")
            batch = data.batch(step, rank, args.batch_size)
            torch.cuda.synchronize(device)
            tick = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = engine(batch, global_step=step)
            finite = torch.isfinite(loss).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite):
                raise RuntimeError("Nonfinite training loss on at least one rank")
            engine.backward(loss)
            engine.step()
            norm = float(engine.get_global_grad_norm())
            if not math.isfinite(norm) or getattr(engine.optimizer, "overflow", False):
                raise RuntimeError("Nonfinite gradient or skipped optimizer update")
            if args.mode == "probe" or step % 25 == 0 or step + 1 == stop:
                if not finite_parameters(parameters):
                    raise RuntimeError("Nonfinite updated parameters")
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - tick
            slowest = torch.tensor(elapsed, device=device)
            dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
            elapsed = float(slowest)
            step_times.append(elapsed)
            step += 1
            row = average_metrics(loss, metrics)
            row.update(step=step, gradient_norm=norm, lr=optimizer.param_groups[0]["lr"],
                       seconds=elapsed, global_batch=world * args.batch_size,
                       samples_per_second=world * args.batch_size / elapsed)
            report["completed_steps"] = step
            if rank == 0:
                write_json(output / "metrics" / f"step_{step:06d}.json", row)
                print("[train] " + json.dumps(row), flush=True)
            if args.mode == "train" and (step % 25 == 0 or step == stop):
                evaluate(step)
            if args.mode == "train" and (step % 100 == 0 or step == stop):
                save(step)
        updated = not torch.equal(initial, parameter_sample(parameters))
        if step_times and not updated:
            raise RuntimeError("No parameter update detected")
        report.update(status="passed", parameters_updated=updated,
                      mean_step_seconds=sum(step_times[1:] or step_times) / max(1, len(step_times[1:] or step_times)))
        stage("complete")
        dist.barrier()
        dist.destroy_process_group()
        return 0
    except Exception as exc:
        report.update(status="cuda_oom" if "out of memory" in str(exc).lower() else "error", error=str(exc))
        stage(report["stage"])
        traceback.print_exc()
        return 1


def search_capacity(run_candidate, limit=64):
    """Find the largest successful integer micro-batch inside a bounded search."""
    low, upper = 0, None
    candidate = 1
    while candidate <= limit:
        status = run_candidate(candidate)
        if status == "passed":
            low = candidate
            if low == limit:
                return low, None
            candidate = min(candidate * 2, limit)
        elif status == "cuda_oom":
            upper = candidate
            break
        else:
            raise RuntimeError(f"Batch {candidate} failed for a non-OOM reason")
    if not low:
        raise RuntimeError("Micro-batch 1 did not complete training")
    while upper is not None and upper - low > 1:
        middle = (low + upper) // 2
        status = run_candidate(middle)
        if status == "passed":
            low = middle
        elif status == "cuda_oom":
            upper = middle
        else:
            raise RuntimeError(f"Batch {middle} failed for a non-OOM reason")
    return low, upper


def launch(args, directory, mode, batch_size, steps, stop_at=None):
    directory.mkdir(parents=True, exist_ok=True)
    # Never let a previous rank's success survive a failed restart.
    for rank in range(8):
        write_json(directory / f"rank_{rank:02d}.json", {"status": "pending", "rank": rank})
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
               str(Path(__file__).resolve()), "worker", "--data", args.data, "--output", str(directory),
               "--mode", mode, "--batch-size", str(batch_size), "--steps", str(steps), "--lr", str(args.lr)]
    if stop_at:
        command += ["--stop-at", str(stop_at)]
    # A timeout is a failure, never evidence of CUDA OOM.
    log_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"cogact_{mode}_{batch_size}_{time.time_ns()}.log"
    with log_path.open("w") as stream:
        process = subprocess.Popen(["timeout", "--kill-after=60", str(args.timeout), *command], stdout=stream, stderr=subprocess.STDOUT)
        hardware = []
        last_progress = 0.0
        while process.poll() is None:
            query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
            if query.returncode == 0:
                hardware.append({"time": time.time(), "gpus": query.stdout.strip().splitlines()})
            if time.monotonic() - last_progress >= 30:
                with log_path.open("rb") as reader:
                    reader.seek(max(0, log_path.stat().st_size - 2048))
                    tail = reader.read().decode(errors="replace").splitlines()[-3:]
                print(f"[progress] {mode} micro_batch={batch_size}: " + " | ".join(tail), flush=True)
                last_progress = time.monotonic()
            time.sleep(1)
        returncode = process.wait()
    shutil.copy2(log_path, directory / f"launcher_{time.time_ns()}.log")
    write_json(directory / f"hardware_{time.time_ns()}.json", hardware)
    reports = [json.loads((directory / f"rank_{rank:02d}.json").read_text()) for rank in range(8)]
    hardware_peaks = {}
    for measurement in hardware:
        for line in measurement["gpus"]:
            fields = [field.strip() for field in line.split(",")]
            gpu, used, total = int(fields[0]), float(fields[1]), float(fields[2])
            hardware_peaks[gpu] = {"used_gib": max(used / 1024, hardware_peaks.get(gpu, {}).get("used_gib", 0)),
                                   "total_gib": total / 1024}
    if returncode == 0 and all(row["status"] == "passed" for row in reports):
        status = "passed"
    elif returncode not in (124, 137) and any(row["status"] == "cuda_oom" for row in reports):
        status = "cuda_oom"
    else:
        status = "error"
    write_json(directory / "summary.json", {"status": status, "returncode": returncode,
                                            "micro_batch": batch_size, "global_batch": batch_size * 8,
                                            "hardware_peak_by_gpu": hardware_peaks, "ranks": reports})
    print(f"[candidate] {mode} micro_batch={batch_size} global_batch={batch_size * 8} status={status}", flush=True)
    if status == "error":
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-60:]), flush=True)
    return status


def sweep(args):
    output = Path(args.output)
    write_json(output / "status.json", {"status": "running", "phase": "capacity"})
    try:
        data = Windows(args.data)
        signature = {"data": data.digest, "lr": args.lr, "world_size": 8,
                     "model": sha256(ROOT / "configs/model/wam.yaml"),
                     "overrides": sha256(ROOT / "configs/smoke/cogact_5080.yaml"),
                     "probe": sha256(Path(__file__)), "train_steps": args.train_steps}
        previous_signature = output / "probe_signature.json"
        if previous_signature.exists() and json.loads(previous_signature.read_text()) != signature:
            raise ValueError("Existing probe output has incompatible code, data or configuration")
        write_json(previous_signature, signature)
        for entry in data.entries:
            if sha256(data.root / entry["path"]) != entry["sha256"]:
                raise ValueError(f"Input checksum mismatch: {entry['path']}")
        def candidate(batch):
            directory = output / "capacity" / f"batch_{batch:03d}"
            previous = directory / "summary.json"
            if previous.exists():
                status = json.loads(previous.read_text())["status"]
                if status in {"passed", "cuda_oom"}:
                    return status
            return launch(args, directory, "probe", batch, 3)
        maximum, failing = search_capacity(candidate, args.max_batch)
        # Leave headroom for evaluation, checkpointing and less favorable inputs.
        training_batch = max(1, min(4, maximum // 2))
        capacity = {"max_tested_micro_batch": maximum, "max_tested_global_batch": maximum * 8,
                    "first_failing_micro_batch": failing, "bounded_by_search_limit": failing is None,
                    "train_micro_batch": training_batch, "world_size": 8,
                    "zero_stage": 2, "gradient_accumulation": 1, "precision": "bf16"}
        write_json(output / "capacity.json", capacity)
        import csv
        capacity_rows = []
        for path in sorted((output / "capacity").glob("batch_*/summary.json")):
            trial = json.loads(path.read_text())
            row = {key: trial[key] for key in ("micro_batch", "global_batch", "status")}
            for field in ("peak_allocated_gib", "peak_reserved_gib", "mean_step_seconds"):
                row[field] = max((rank.get(field, 0) for rank in trial["ranks"]), default=0)
            row["hardware_peak_used_gib"] = max((v["used_gib"] for v in trial["hardware_peak_by_gpu"].values()), default=0)
            capacity_rows.append(row)
        with (output / "capacity.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(capacity_rows[0]))
            writer.writeheader()
            writer.writerows(capacity_rows)
        write_json(output / "status.json", {"status": "running", "phase": "loss_trend"})
        train_output = output / "training"
        # A real restart verifies optimizer, scheduler, RNG and step restoration.
        halfway = max(1, args.train_steps // 2)
        for stop in (halfway, args.train_steps):
            if launch(args, train_output, "train", training_batch, args.train_steps, stop) != "passed":
                raise RuntimeError("Training/resume phase failed; inspect training logs")
        final_ranks = [json.loads((train_output / f"rank_{rank:02d}.json").read_text()) for rank in range(8)]
        if not all(row.get("resumed_from_step", 0) >= halfway for row in final_ranks):
            raise RuntimeError("The second training process did not restore its checkpoint")
        evaluations = [json.loads(p.read_text()) for p in sorted((train_output / "eval").glob("step_*.json"))]
        metrics = [json.loads(p.read_text()) for p in sorted((train_output / "metrics").glob("step_*.json"))]
        for filename, rows in (("training.csv", metrics), ("evaluation.csv", evaluations)):
            with (output / filename).open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        trend = {}
        for key in ("loss_total", "loss_action", "loss_video", "loss_future_delta", "loss_semantic_future"):
            start, end = evaluations[0][key], evaluations[-1][key]
            slope = float(np.polyfit([row["step"] for row in evaluations], [row[key] for row in evaluations], 1)[0])
            trend[key] = {"initial": start, "final": end, "slope_per_step": slope,
                          "decreased": end < start and slope < 0}
        write_json(output / "result.json", {"status": "passed", "capacity": capacity, "loss_trend": trend,
                   "resume_verified": True,
                   "scope": "real single-segment windows; full online model; no generalization claim"})
        write_json(output / "status.json", {"status": "passed", "phase": "complete"})
        return 0
    except Exception as exc:
        write_json(output / "status.json", {"status": "error", "error": str(exc)})
        traceback.print_exc()
        return 1


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--sample-dir", default=".cache/cogact_smoke")
    p.add_argument("--output", default=".cache/cogact_a100/data")
    p.add_argument("--windows", type=int, default=64)
    for command in ("worker", "sweep"):
        p = commands.add_parser(command)
        p.add_argument("--data", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--lr", type=float, default=5e-5)
        if command == "worker":
            p.add_argument("--mode", choices=("probe", "train"), required=True)
            p.add_argument("--batch-size", type=int, required=True)
            p.add_argument("--steps", type=int, required=True)
            p.add_argument("--stop-at", type=int)
        else:
            p.add_argument("--max-batch", type=int, default=64)
            p.add_argument("--train-steps", type=int, default=200)
            p.add_argument("--timeout", type=int, default=14400)
    args = parser.parse_args()
    return {"prepare": prepare, "worker": worker, "sweep": sweep}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
