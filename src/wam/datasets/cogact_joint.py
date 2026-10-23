"""CogACT joint-only projection into InternW0's masked 80D robot layout.

InternW0 uses absolute targets for joint/gripper slots; only the EEF pose
slots use deltas. See configs/pretrain/embodiments/agibot.yaml and
configs/pretrain/projections/robot.yaml. Missing EEF/body channels are masked.
The source format stores 56 float32 values per frame without a NumPy header.
"""

from __future__ import annotations

import numpy as np
import torch

JOINT_INDICES = [*range(7), 16, *range(40, 47), 56]


def compact_state(records: np.ndarray) -> torch.Tensor:
    """Current measured joints and source gripper state scaled by 120."""
    rows = np.asarray(records)
    if rows.ndim != 2 or rows.shape[1] != 56:
        raise ValueError(f"Expected [T,56] source records, got {rows.shape}")
    return torch.from_numpy(np.concatenate([
        rows[:, 38:45], rows[:, 54:55] / 120.0,
        rows[:, 45:52], rows[:, 55:56] / 120.0,
    ], axis=-1).astype(np.float32))


def compact_action(current: np.ndarray, following: np.ndarray) -> torch.Tensor:
    """Absolute next measured joint targets, with current gripper commands.

This realizes InternW0's supported action_from_next_state convention. For
the local smoke episode, next commanded and next measured joints are also
audited separately; equivalence is not assumed for other recordings.
"""
    if current.shape != following.shape or current.ndim != 2 or current.shape[1] != 56:
        raise ValueError("Current and following source records must both be [T,56]")
    return torch.from_numpy(np.concatenate([
        following[:, 38:45], current[:, 20:21],
        following[:, 45:52], current[:, 21:22],
    ], axis=-1).astype(np.float32))


def scatter80(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 80D values and an exclusion mask (True means unused)."""
    if values.shape[-1] != 16:
        raise ValueError(f"Expected 16 compact channels, got {values.shape}")
    output = values.new_zeros((*values.shape[:-1], 80))
    output[..., JOINT_INDICES] = values
    excluded = torch.ones(80, dtype=torch.bool, device=values.device)
    excluded[JOINT_INDICES] = False
    return output, excluded


def sample_joint_window(
    records: np.ndarray, *, start: int, end: int, current: int, horizon: int = 32,
) -> dict[str, torch.Tensor]:
    """Sample within a half-open segment, excluding every padded action.

State[k] is measured at t+k; action[k] targets t+k+1. Padding repeats the
last record, but receives no loss. No OpenPi-specific terminal bonus steps.
"""
    if not (0 <= start <= current < end - 1 <= len(records) - 1) or horizon < 1:
        raise ValueError("Invalid segment, current frame, or horizon")
    state_idx = current + np.arange(horizon + 1)
    action_idx = state_idx[:-1]
    following_idx = state_idx[1:]
    state = compact_state(records[np.clip(state_idx, start, end - 1)])
    action = compact_action(
        records[np.clip(action_idx, start, end - 1)],
        records[np.clip(following_idx, start, end - 1)],
    )
    return {
        "state": state,
        "action": action,
        "state_is_pad": torch.from_numpy(state_idx >= end),
        "action_is_pad": torch.from_numpy(following_idx >= end),
    }
