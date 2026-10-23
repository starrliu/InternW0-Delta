from pathlib import Path

import av
import numpy as np
import pytest

from wam.datasets.cogact_video import IndexedVideoReader


def make_video(path: Path, frames=40):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = 32, 24
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "g": "8", "bf": "0"}
        for i in range(frames):
            rgb = np.full((24, 32, 3), (i * 6) % 256, np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        decoded = list(container.decode(stream))
    stat = path.stat()
    np.savez(str(path) + ".frames.npz", pts=np.array([f.pts for f in decoded], np.int64),
             time_base=np.array([tb.numerator, tb.denominator], np.int64),
             size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return [f.to_ndarray(format="rgb24") for f in decoded]


def test_exact_noncontiguous_and_repeated_frame_reads(tmp_path):
    path = tmp_path / "frames.mp4"
    reference = make_video(path)
    reader = IndexedVideoReader(cache_size=1)
    indices = [39, 0, 24, 28, 32, 24]
    actual = reader.read(path, indices)
    for image, index in zip(actual, indices):
        np.testing.assert_array_equal(image, reference[index])
    with pytest.raises(IndexError):
        reader.read(path, [40])
    with pytest.raises(IndexError):
        reader.read(path, [-1])


def test_stale_sidecar_rejected_without_rebuild(tmp_path):
    path = tmp_path / "frames.mp4"
    make_video(path, frames=2)
    sidecar = Path(str(path) + ".frames.npz")
    before = sidecar.read_bytes()
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Stale"):
        IndexedVideoReader().read(path, [0])
    assert sidecar.read_bytes() == before
