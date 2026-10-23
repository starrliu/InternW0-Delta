"""Prepared, read-only CogACT data for the official InternW0 frame trainer."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cogact_joint import JOINT_INDICES, compact_state, sample_joint_window, scatter80
from .cogact_video import IndexedVideoReader, VIEWS, canvas
from .lerobot.robot_video_dataset import DEFAULT_PROMPT
from .lerobot.utils.normalizer import SingleFieldLinearNormalizer


class CogACTDataset(Dataset):
    def __init__(self, prepared_dir, text_embedding_cache_dir, is_training_set=True,
                 num_frames=33, action_video_freq_ratio=4, video_size=(384, 256),
                 memory_video_anchor_size=1, memory_recent_frame_offset=32,
                 context_len=128, processor=None, dataset_dirs=None,
                 pretrained_norm_stats=None, load_text_context=True):
        if (num_frames, action_video_freq_ratio, tuple(video_size), memory_video_anchor_size,
                memory_recent_frame_offset, context_len) != (33, 4, (384, 256), 1, 32, 128):
            raise ValueError("CogACT adapter currently requires the verified 32-step, 9-frame recipe")
        self.prepared_dir = Path(prepared_dir)
        marker = json.loads((self.prepared_dir / "complete.json").read_text())
        for name in ("manifest.json", "normalization.json", "episodes.jsonl", "action_audit.json"):
            if hashlib.sha256((self.prepared_dir / name).read_bytes()).hexdigest() != marker["files"][name]:
                raise ValueError("Prepared data digest mismatch: " + name)
        self.manifest = json.loads((self.prepared_dir / "manifest.json").read_text())
        stats = json.loads((self.prepared_dir / "normalization.json").read_text())
        if stats["signature"] != marker["signature"]:
            raise ValueError("Normalization and dataset index signatures differ")
        if pretrained_norm_stats and Path(pretrained_norm_stats).resolve() != (self.prepared_dir / "normalization.json").resolve():
            raise ValueError("Use the statistics belonging to this prepared dataset")
        self.is_training_set = is_training_set
        split = "train" if is_training_set else "validation"
        self.episodes = [row for line in (self.prepared_dir / "episodes.jsonl").read_text().splitlines()
                         if (row := json.loads(line))["split"] == split]
        if not self.episodes:
            raise ValueError(f"No samples in split {split}")
        self.ends = np.cumsum([row["sample_count"] for row in self.episodes], dtype=np.int64)
        self.sample_episode_ranges = tuple(zip([0, *self.ends[:-1].tolist()], self.ends.tolist()))
        self.root = Path(self.manifest["dataset_root"])
        self.lengths = {(row["source_index"], row["video_path"]): row["bytes"] // (56 * 4)
                        for row in json.loads((self.prepared_dir / "action_audit.json").read_text())}
        self.state_norm = SingleFieldLinearNormalizer(
            {k: torch.tensor(v, dtype=torch.float32) for k, v in stats["state"].items() if k != "count"}, "z-score")
        self.action_norm = SingleFieldLinearNormalizer(
            {k: torch.tensor(v, dtype=torch.float32) for k, v in stats["action"].items() if k != "count"}, "z-score")
        self.text_embedding_cache_dir = Path(text_embedding_cache_dir)
        self.load_text_context = load_text_context
        self.signature = marker["signature"]
        self._reset_caches()

    def _reset_caches(self):
        self._pid = os.getpid()
        self._records, self._contexts = OrderedDict(), OrderedDict()
        self._split_lengths = {}
        self._reader = IndexedVideoReader()

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_reader", "_records", "_contexts", "_split_lengths", "_pid"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_caches()

    def __len__(self):
        return int(self.ends[-1])

    def denormalize_action(self, action, **kwargs):
        values = action[..., JOINT_INDICES].to(device="cpu", dtype=torch.float32)
        return scatter80(self.action_norm.backward(values))[0]

    def _context(self, prompt):
        key = hashlib.sha256(prompt.encode()).hexdigest()
        saved = self._contexts.pop(key, None)
        if saved is None:
            path = self.text_embedding_cache_dir / f"{key}.t5_len128.wan22ti2v5b.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            context, mask = payload["context"], payload["mask"].bool()
            if context.shape != (128, 4096) or mask.shape != (128,) or not torch.isfinite(context).all():
                raise ValueError("Invalid text context: " + str(path))
            saved = context.clone().masked_fill(~mask[:, None], 0), torch.ones_like(mask)
        self._contexts[key] = saved
        while len(self._contexts) > 32:
            self._contexts.popitem(last=False)
        return saved

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if getattr(self, "_pid", None) != os.getpid():
            self._reset_caches()
        episode_index = int(np.searchsorted(self.ends, index, side="right"))
        episode = self.episodes[episode_index]
        offset = index - (0 if episode_index == 0 else int(self.ends[episode_index - 1]))
        start, end = episode["start"], episode["end"]
        current = start + offset
        source_index, video_path = episode["source_index"], episode["video_path"]
        source = self.manifest["datasets"][source_index]
        folder = self.root / source["dataset_folder"]
        key = source_index, video_path
        records = self._records.pop(key, None)
        if records is None:
            path = folder / source.get("action_folder_name", "actions_cmd_rfix_v1") / video_path / "action.npy"
            length = self.lengths[key]
            if path.stat().st_size != length * 56 * 4:
                raise ValueError("Source action size changed since preparation")
            records = np.memmap(path, dtype=np.float32, mode="r", shape=(length, 56))
        self._records[key] = records
        while len(self._records) > 32:
            self._records.popitem(last=False)
        if source_index not in self._split_lengths:
            video_manifest = json.loads((folder / "indexed_mp4/manifest.json").read_text())
            if video_manifest.get("complete") is not True or video_manifest.get("format") != "agibot-indexed-mp4-v2":
                raise ValueError("Incomplete video preparation")
            self._split_lengths[source_index] = video_manifest["split_length"]
        raw = sample_joint_window(records, start=start, end=end, current=current)
        action, action_mask = scatter80(self.action_norm.forward(raw["action"]))
        state, state_mask = scatter80(self.state_norm.forward(raw["state"]))
        recent = max(start, current - 32)
        future = current + np.arange(0, 33, 4)
        indices = np.concatenate([[start, recent], np.clip(future, start, end - 1)])
        views = self._reader.views(folder, video_path, indices, self._split_lengths[source_index])
        memory, _ = scatter80(self.state_norm.forward(compact_state(records[[start, recent]])))
        prompt = DEFAULT_PROMPT.format(task=episode["task"])
        result = {
            "sample_id": index, "video": canvas(views[:, 2:]), "action": action,
            "proprio": state[:-1], "prompt": prompt,
            "video_layout": "single", "video_view_names": "canvas",
            "video_canvas_layout": "robotwin", "video_canvas_view_names": "|".join(VIEWS),
            "vlm_current_images": (views[:, 2] * 255).round().clamp(0, 255).to(torch.uint8),
            "image_is_pad": torch.from_numpy(future >= end),
            "action_is_pad": raw["action_is_pad"], "action_dim_is_pad": action_mask,
            "proprio_is_pad": raw["state_is_pad"][:-1], "proprio_dim_is_pad": state_mask,
        }
        for name, position, padded in (("anchor", 0, False), ("recent", 1, current - 32 < start)):
            prefix = "memory_video_" + name
            result.update({prefix: canvas(views[:, position:position + 1]),
                           prefix + "_is_pad": torch.tensor([padded]),
                           prefix + "_proprio": memory[position:position + 1],
                           prefix + "_proprio_is_pad": torch.tensor([padded])})
        if self.load_text_context:
            result["context"], result["context_mask"] = self._context(prompt)
        return result
