"""Read immutable AgiBot indexed MP4s by their exact presentation timestamps.

The existing .frames.npz sidecars are required. This reader never rebuilds an
index, writes a cache beside a video, or substitutes a different frame.
"""
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TF

VIEWS = ("head_color", "hand_left_color", "hand_right_color")


class IndexedVideoReader:
    def __init__(self, cache_size=64):
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def _index(self, path):
        stat = path.stat()
        fingerprint = stat.st_size, stat.st_mtime_ns
        saved = self.cache.pop(str(path), None)
        if saved is not None and saved[0] == fingerprint:
            self.cache[str(path)] = saved
            return saved[1:]
        with np.load(str(path) + ".frames.npz", allow_pickle=False) as index:
            if (int(index["size"]), int(index["mtime_ns"])) != fingerprint:
                raise ValueError(f"Stale video index: {path}")
            pts = index["pts"].copy()
            time_base = tuple(map(int, index["time_base"]))
        if pts.ndim != 1 or pts.dtype != np.int64 or not len(pts) or np.any(np.diff(pts) <= 0):
            raise ValueError(f"Invalid frame timestamps: {path}")
        self.cache[str(path)] = (fingerprint, pts, time_base)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return pts, time_base

    def read(self, path, indices):
        import av
        path = Path(path)
        pts, time_base = self._index(path)
        requested = sorted(set(map(int, indices)))
        if not requested or requested[0] < 0 or requested[-1] >= len(pts):
            raise IndexError(f"Requested frame outside video: {path}")
        # Seek separately for distant history; decode adjacent future observations
        # in one pass. Repeated clipped frames are decoded only once.
        groups = []
        for index in requested:
            if not groups or index - groups[-1][-1] > 16:
                groups.append([])
            groups[-1].append(index)
        images = {}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            if (stream.time_base.numerator, stream.time_base.denominator) != time_base:
                raise ValueError(f"Video time base differs from index: {path}")
            for group in groups:
                targets = {int(pts[i]): i for i in group}
                container.seek(int(pts[group[0]]), stream=stream, backward=True, any_frame=False)
                for frame in container.decode(stream):
                    if frame.pts in targets:
                        images[targets.pop(frame.pts)] = frame.to_ndarray(format="rgb24")
                    if not targets:
                        break
                    if frame.pts is None or frame.pts > int(pts[group[-1]]):
                        break
                if targets:
                    raise RuntimeError(f"Decoder missed exact frames: {path}")
        return [images[int(i)] for i in indices]

    def views(self, dataset_root, video_path, indices, split_length):
        root = Path(dataset_root) / "indexed_mp4/videos_h264" / video_path / "videos"
        outputs = []
        for view in VIEWS:
            groups = defaultdict(list)
            for position, index in enumerate(indices):
                part, local = (None, int(index)) if split_length is None else divmod(int(index), split_length)
                groups[part].append((position, local))
            ordered = [None] * len(indices)
            for part, positions in groups.items():
                name = view + ("" if part is None else f"_{part}") + ".mp4"
                frames = self.read(root / name, [local for _, local in positions])
                for (position, _), frame in zip(positions, frames):
                    if view == "head_color" and video_path.split("/", 1)[0] == "3173":
                        if frame.shape[0] == 800:
                            frame = frame[40:760]
                        elif frame.shape[0] == 500:
                            frame = frame[25:475]
                    tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255
                    ordered[position] = TF.resize(tensor, [192, 256], antialias=True)
            outputs.append(torch.stack(ordered))
        return torch.stack(outputs)


def canvas(views):
    head = TF.resize(views[0], [256, 256], antialias=True)
    left = TF.resize(views[1], [128, 128], antialias=True)
    right = TF.resize(views[2], [128, 128], antialias=True)
    image = torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2)
    return (image * 2 - 1).permute(1, 0, 2, 3).contiguous()
