import numpy as np
import pytest
import torch

from wam.datasets.cogact_joint import JOINT_INDICES, compact_action, sample_joint_window, scatter80


def records():
    rows = np.zeros((10, 56), dtype=np.float32)
    for t in range(10):
        rows[t, 2:16] = -1000 - t  # Must not silently substitute commanded fields.
        rows[t, 38:52] = t * 100 + np.arange(14)
        rows[t, 20:22] = [t / 10, 1 - t / 10]
        rows[t, 54:56] = [t * 12, 120 - t * 12]
    return rows


def test_absolute_next_joints_current_gripper_and_state():
    sample = sample_joint_window(records(), start=2, end=9, current=4, horizon=3)
    assert sample["state"][0, 0] == 400
    assert sample["state"][0, 7] == pytest.approx(0.4)
    assert sample["action"][0, 0] == 500  # Absolute, not a delta of 100.
    assert sample["action"][0, 8] == 507
    assert sample["action"][0, 7] == pytest.approx(0.4)  # t, not t+1.
    assert sample["action"][0, 15] == pytest.approx(0.6)


def test_padding_never_reads_next_segment_or_adds_terminal_loss():
    sample = sample_joint_window(records(), start=2, end=6, current=4, horizon=4)
    assert sample["action_is_pad"].tolist() == [False, True, True, True]
    assert sample["state_is_pad"].tolist() == [False, False, True, True, True]
    assert sample["action"][:, 0].tolist() == [500] * 4


def test_scatter_and_inverse_preserve_only_canonical_joint_slots():
    values = compact_action(records()[:3], records()[1:4])
    output, excluded = scatter80(values)
    assert output.shape == (3, 80)
    assert (~excluded).sum().item() == 16
    torch.testing.assert_close(output[:, JOINT_INDICES], values)
    assert torch.count_nonzero(output[:, excluded]) == 0


@pytest.mark.parametrize("start,end,current", [(0, 11, 1), (2, 6, 1), (2, 6, 5)])
def test_invalid_windows_fail(start, end, current):
    with pytest.raises(ValueError):
        sample_joint_window(records(), start=start, end=end, current=current)
