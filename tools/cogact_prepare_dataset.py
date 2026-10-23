#!/usr/bin/env python3
"""Audit read-only CogACT sources and write an InternW0 dataset index/statistics.

An epoch visits every eligible frame start once (stride=1). The final frame of
each segment is excluded because an action needs the next measured state.
Normalization uses each valid transition once, not padded or repeated horizons.
Entire recordings share a deterministic split even when present in two sources.
Only the output directory is written; no source statistics or indexes are built.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def recording_split(video_path, validation_fraction, seed):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{video_path}".encode()).digest()[:8], "big")
    return "validation" if value / 2**64 < validation_fraction else "train"


class Moments:
    def __init__(self):
        self.count = 0
        self.mean = np.zeros(16, np.float64)
        self.m2 = np.zeros(16, np.float64)
        self.minimum = np.full(16, np.inf)
        self.maximum = np.full(16, -np.inf)

    def add(self, values):
        values = np.asarray(values, np.float64)
        if values.ndim != 2 or values.shape[1] != 16 or not np.isfinite(values).all():
            raise ValueError("Invalid/nonfinite joint-gripper values")
        n = len(values)
        if not n:
            return
        mean = values.mean(0)
        delta = mean - self.mean
        total = self.count + n
        self.m2 += ((values - mean)**2).sum(0) + delta**2 * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total
        self.minimum = np.minimum(self.minimum, values.min(0))
        self.maximum = np.maximum(self.maximum, values.max(0))

    def result(self):
        if self.count == 0:
            raise ValueError("No training rows for normalization")
        return {"count": self.count, "mean": self.mean.tolist(),
                "std": np.sqrt(np.maximum(self.m2 / self.count, 0)).tolist(),
                "min": self.minimum.tolist(), "max": self.maximum.tolist()}


def state16(rows):
    return np.concatenate([rows[:, 38:45], rows[:, 54:55] / 120,
                           rows[:, 45:52], rows[:, 55:56] / 120], axis=1)


def action16(current, following):
    return np.concatenate([following[:, 38:45], current[:, 20:21],
                           following[:, 45:52], current[:, 21:22]], axis=1)


def below(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes its declared root")
    return path


def prepare(manifest_path, output, validation_fraction=0, seed=42, require_images=True):
    started = time.monotonic()
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["dataset_root"]).resolve()
    output = Path(output).resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output must be separate from the read-only data source")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must lie in [0,1)")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; existing preparations are immutable")
    output.mkdir(parents=True, exist_ok=True)
    state, action = Moments(), Moments()
    episodes, datasets, file_audit, snapshot = [], [], [], []
    recording_intervals = defaultdict(list)
    split_counts = Counter()
    prompts = set()
    for source_index, source in enumerate(manifest["datasets"]):
        if float(source.get("weight", 1)) != 1:
            raise ValueError("Preparation currently supports weight=1 sources only")
        folder = below(root, source["dataset_folder"])
        metadata_path = below(folder, source["metadata_file"])
        action_root = below(folder, source.get("action_folder_name", "actions_cmd_rfix_v1"))
        action_metadata_path = action_root / "meta_data.json"
        metadata = np.load(metadata_path, allow_pickle=True).item()
        action_metadata = json.loads(action_metadata_path.read_text())
        required = ("video_path", "instructions", "start_end", "quality")
        n = len(metadata["video_path"])
        if any(len(metadata[key]) != n for key in required):
            raise ValueError("Inconsistent episode metadata columns")
        image_records = None
        image_manifest = None
        if require_images:
            image_manifest_path = folder / "indexed_mp4/manifest.json"
            image_manifest = json.loads(image_manifest_path.read_text())
            if image_manifest.get("format") != "agibot-indexed-mp4-v2" or image_manifest.get("complete") is not True:
                raise ValueError("Incomplete indexed MP4 source: " + str(folder))
            inventory_path = folder / "indexed_mp4/files.json"
            inventory_digest = sha256(inventory_path)
            if inventory_digest != image_manifest["files_sha256"]:
                raise ValueError("Video inventory digest mismatch")
            image_records = {x["path"]: x for x in json.loads(inventory_path.read_text())}
            snapshot.extend([{"path": str(image_manifest_path), "sha256": sha256(image_manifest_path)},
                             {"path": str(inventory_path), "sha256": inventory_digest}])
        snapshot.extend({"path": str(p), "sha256": sha256(p)} for p in (metadata_path, action_metadata_path))
        selected = defaultdict(list)
        for i in range(n):
            if str(metadata["quality"][i]) not in ("good", "medium"):
                continue
            start, end = map(int, metadata["start_end"][i])
            if end - start < 2:
                continue
            video = str(metadata["video_path"][i])
            task = str(metadata["instructions"][i]).strip().lower()
            if not task:
                raise ValueError("Empty instruction")
            split = recording_split(video, validation_fraction, seed)
            row = {"source_index": source_index, "dataset_folder": source["dataset_folder"],
                   "episode_index": i, "video_path": video, "start": start, "end": end,
                   "task": task, "quality": str(metadata["quality"][i]), "split": split,
                   "sample_count": end - start - 1}
            selected[video].append(row)
            episodes.append(row)
            recording_intervals[video].append((source_index, start, end))
            split_counts[split] += row["sample_count"]
            prompts.add(task)
        for video, rows in sorted(selected.items()):
            info = action_metadata[video]
            length = int(info["length"])
            if sum(info["dim_list"]) != 56:
                raise ValueError("Expected a 56D raw action file")
            path = below(action_root, video + "/action.npy")
            before = path.stat()
            if before.st_size != length * 56 * 4:
                raise ValueError("Action file size mismatch: " + str(path))
            records = np.memmap(path, dtype=np.float32, mode="r", shape=(length, 56))
            max_difference = 0.0
            for row in rows:
                start, end = row["start"], row["end"]
                if not 0 <= start < end <= length:
                    raise ValueError("Episode outside action bounds")
                segment = records[start:end]
                s, a = state16(segment), action16(segment[:-1], segment[1:])
                if not np.isfinite(s).all() or not np.isfinite(a).all():
                    raise ValueError("Nonfinite active channels: " + str(path))
                if row["split"] == "train":
                    state.add(s)
                    action.add(a)
                difference = np.abs(segment[:, 2:16] - segment[:, 38:52])
                if not np.isfinite(difference).all():
                    raise ValueError("Nonfinite commanded/measured joint comparison")
                max_difference = max(max_difference, float(difference.max()))
                if image_records is not None:
                    split_length = image_manifest["split_length"]
                    parts = [None] if split_length is None else range(start // split_length, (end - 1) // split_length + 1)
                    for part in parts:
                        for view in ("head_color", "hand_left_color", "hand_right_color"):
                            suffix = "" if part is None else f"_{part}"
                            key = f"videos_h264/{video}/videos/{view}{suffix}.mp4"
                            entry = image_records.get(key)
                            if entry is None:
                                raise ValueError("Missing video in inventory: " + key)
                            needed = end if part is None else min(end - part * split_length, split_length)
                            if int(entry["validation"]["frames"]) < needed:
                                raise ValueError("Video does not cover episode: " + key)
            checksum = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Source action file changed during preparation")
            file_audit.append({"source_index": source_index, "video_path": video,
                               "bytes": before.st_size, "sha256": checksum,
                               "command_vs_measured_joint_max_abs": max_difference})
            del records
        counts = Counter()
        for rows in selected.values():
            for row in rows:
                counts[row["split"]] += row["sample_count"]
        summary = {"dataset_folder": source["dataset_folder"], "metadata_file": source["metadata_file"],
                   "episodes": sum(map(len, selected.values())), "recordings": len(selected),
                   "train_samples": counts["train"], "validation_samples": counts["validation"]}
        datasets.append(summary)
        print(json.dumps(summary), flush=True)
    # Multiple appearances of the same recording stay in one split. Report overlap
    # rather than silently changing the dataset mixture or its sample weighting.
    overlaps = []
    for video, intervals in recording_intervals.items():
        ordered = sorted(intervals, key=lambda x: x[1])
        for i, first in enumerate(ordered):
            for second in ordered[i + 1:]:
                if second[1] >= first[2]:
                    break
                if first[0] != second[0]:
                    overlaps.append({"video_path": video, "a": first, "b": second})
    signature = hashlib.sha256(json.dumps({"manifest": manifest, "sources": snapshot,
        "actions": file_audit, "validation_fraction": validation_fraction, "seed": seed}, sort_keys=True).encode()).hexdigest()
    norm = {"version": 1, "mode": "z-score", "signature": signature,
            "scope": "training recordings only; raw transitions once per selected segment",
            "compact_order": "left joints 7, left gripper, right joints 7, right gripper",
            "state": state.result(), "action": action.result()}
    report = {"status": "prepared", "signature": signature, "dataset_root": str(root),
              "validation_fraction": validation_fraction, "seed": seed, "horizon": 32,
              "sample_stride": 1, "terminal_repeat_valid_steps": 0,
              "counts": dict(split_counts), "datasets": datasets,
              "cross_source_overlap_count": len(overlaps), "cross_source_overlaps": overlaps,
              "video_check": "inventory coverage only; decode/read benchmark pending",
              "command_vs_measured_joint_max_abs": max(x["command_vs_measured_joint_max_abs"] for x in file_audit),
              "elapsed_seconds": time.monotonic() - started}
    for name, payload in (("normalization.json", norm), ("report.json", report),
                          ("manifest.json", manifest), ("source_fingerprints.json", snapshot),
                          ("action_audit.json", file_audit), ("tasks.json", sorted(prompts))):
        (output / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with (output / "episodes.jsonl").open("w") as stream:
        for row in episodes:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Reuse the official text-cache command without making the source look like
    # LeRobot or writing derived task metadata into the original dataset folders.
    (output / "meta").mkdir()
    with (output / "meta/tasks.jsonl").open("w") as stream:
        for i, task in enumerate(sorted(prompts)):
            stream.write(json.dumps({"task_index": i, "task": task}, ensure_ascii=False) + "\n")
    (output / "complete.json").write_text(json.dumps({"signature": signature, "files": {
        str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}}) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prepare(args.manifest, args.output, args.validation_fraction, args.seed)
    print(json.dumps({"counts": report["counts"], "signature": report["signature"]}), flush=True)
