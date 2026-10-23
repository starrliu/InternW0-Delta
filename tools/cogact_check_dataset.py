#!/usr/bin/env python3
"""Check real prepared samples without GPU use or source writes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from wam.datasets.cogact_dataset import CogACTDataset
from wam.datasets.cogact_joint import JOINT_INDICES, compact_action


def check(prepared, output, text_cache=None):
    torch.set_num_threads(2)
    ds = CogACTDataset(prepared, text_cache or '/unused', load_text_context=text_cache is not None)
    selected = {}
    for i, ep in enumerate(ds.episodes):
        base = 0 if i == 0 else int(ds.ends[i - 1])
        source = ep['source_index']
        selected.setdefault((source, 'start'), base)
        selected.setdefault((source, 'tail'), int(ds.ends[i]) - 1)
        boundary = ((ep['start'] // 500) + 1) * 500
        current = boundary - 8
        if ep['start'] <= current < ep['end'] - 32:
            selected.setdefault((source, 'cross_500'), base + current - ep['start'])
    rows = []
    for (source, kind), index in sorted(selected.items()):
        started = time.perf_counter()
        sample = ds[index]
        elapsed = time.perf_counter() - started
        assert sample['video'].shape == (3, 9, 384, 256)
        assert sample['action'].shape == sample['proprio'].shape == (32, 80)
        assert sample['vlm_current_images'].shape == (3, 3, 192, 256)
        assert int((~sample['action_dim_is_pad']).sum()) == 16
        for name, value in sample.items():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all(), name
        assert sample['video'].min() >= -1 - 1e-6 and sample['video'].max() <= 1 + 1e-6
        if kind == 'tail':
            assert int((~sample['action_is_pad']).sum()) == 1
        ep_index = int(np.searchsorted(ds.ends, index, side='right'))
        ep = ds.episodes[ep_index]
        base = 0 if ep_index == 0 else int(ds.ends[ep_index - 1])
        current = ep['start'] + index - base
        records = ds._records[(source, ep['video_path'])]
        expected = compact_action(records[current:current+1], records[current+1:current+2])
        # Only unclipped values are reversible under the official [-5,5] clamp.
        unclipped = (expected * ds.action_norm.scale + ds.action_norm.offset).abs() < 5
        restored = ds.denormalize_action(sample['action'][:1])[..., JOINT_INDICES]
        torch.testing.assert_close(restored[unclipped], expected[unclipped], atol=2e-6, rtol=2e-5)
        batched = default_collate([sample, sample])
        assert batched['video'].shape == (2, 3, 9, 384, 256)
        rows.append({'source': source, 'kind': kind, 'index': index, 'seconds': elapsed})
        print(json.dumps(rows[-1]), flush=True)
    assert len({source for source, _ in selected}) == len(ds.manifest['datasets'])
    result = {'status': 'passed', 'signature': ds.signature, 'samples_per_epoch': len(ds),
              'episodes': len(ds.episodes), 'text_context_checked': text_cache is not None,
              'checks': rows}
    Path(output).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--text-cache')
    args = parser.parse_args()
    check(args.prepared, args.output, args.text_cache)
