"""Offline repartitioning of rigid, pure-data-parallel DeepSpeed ZeRO-2 states.

Only changes the storage partitions. FP32 master weights, Adam moments, step,
learning-rate scheduler and global data position are preserved. The source must
be quiescent; it is never modified. MoE, tensor parallelism, round-robin ordering
and elastic checkpoints are deliberately rejected.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch


def partition_size(numel, world_size):
    # DeepSpeed aligns the flattened group to 2 * DP world size.
    alignment = 2 * world_size
    return ((numel + alignment - 1) // alignment) * 2


def fragment_map(shapes, rank, size, address_type):
    result = OrderedDict()
    begin, end, offset = rank * size, (rank + 1) * size, 0
    for name, shape in shapes.items():
        count = math.prod(shape)
        lo, hi = max(begin, offset), min(end, offset + count)
        if hi > lo:
            result[name] = address_type(numel=hi-lo, start=lo-begin)
        offset += count
    return result


def slice_partitions(parts, start, length, old_size):
    """Copy a global interval, excluding source padding, without concatenation."""
    result = torch.empty(length, dtype=parts[0].dtype)
    written = 0
    while written < length:
        rank, offset = divmod(start + written, old_size)
        count = min(length-written, parts[rank].numel()-offset)
        if count <= 0:
            raise ValueError('Checkpoint contains a gap in the flattened group')
        result[written:written+count].copy_(parts[rank][offset:offset+count])
        written += count
    return result


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy()).hexdigest()


@torch.no_grad()
def reshard_checkpoint(source, destination, world_size, *, seed=42, validate_only=False):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f'Refuse to overwrite {destination}')
    if world_size < 1:
        raise ValueError('world_size must be positive')
    model_path = source / 'pytorch_model/mp_rank_00_model_states.pt'
    model = torch.load(model_path, mmap=True, map_location='cpu', weights_only=False)
    old_world = int(model['dp_world_size'])
    config = model['ds_config']
    if int(model['mp_world_size']) != 1 or config['zero_optimization'].get('round_robin_gradients', False):
        raise ValueError('Only pure data parallelism without round-robin gradients is supported')
    if int(config['zero_optimization']['stage']) != 2:
        raise ValueError('Only ZeRO stage 2 is supported')
    shapes = model['param_shapes']
    lengths = [sum(math.prod(s) for s in g.values()) for g in shapes]
    files = [source / f'pytorch_model/bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt' for r in range(old_world)]
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in source.rglob('*') if p.is_file()}
    shards = [torch.load(p, mmap=True, map_location='cpu', weights_only=False)['optimizer_state_dict'] for p in files]
    old_sizes = [partition_size(n, old_world) for n in lengths]
    new_sizes = [partition_size(n, world_size) for n in lengths]
    template = shards[0]
    groups = template['base_optimizer_state']['param_groups']
    if len(groups) != len(shapes) or any(len(g['params']) != 1 for g in groups):
        raise ValueError('Expected one flattened parameter per optimizer group')
    address_type = type(next(iter(template['param_slice_mappings'][0].values())))
    for rank, shard in enumerate(shards):
        if list(shard['partition_count']) != [old_world]*len(shapes):
            raise ValueError('MoE / mixed partition counts are unsupported')
        if not isinstance(shard['base_optimizer_state'], dict) or int(shard['zero_stage']) != 2:
            raise ValueError('Expected rigid ZeRO-2 state')
        if shard['base_optimizer_state']['param_groups'] != groups:
            raise ValueError('Optimizer hyperparameters differ between shards')
        for i, n in enumerate(lengths):
            valid = max(0, min(old_sizes[i], n-rank*old_sizes[i]))
            if shard['single_partition_of_fp32_groups'][i].numel() != valid:
                raise ValueError('FP32 partition size mismatch')
            if shard['param_slice_mappings'][i] != fragment_map(shapes[i], rank, old_sizes[i], address_type):
                raise ValueError('Unsupported parameter ordering or fragment map')
            state = shard['base_optimizer_state']['state'][groups[i]['params'][0]]
            if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Expected Adam step and two moment tensors')
            if int(state['step'].item()) != int(model['global_steps']):
                raise ValueError('Adam step does not match model step')
            for key in ['exp_avg', 'exp_avg_sq']:
                if state[key].numel() != old_sizes[i]:
                    raise ValueError('Optimizer partition size mismatch')
    global_batch = int(config['train_batch_size'])
    accum = int(config['gradient_accumulation_steps'])
    if global_batch % (world_size * accum):
        raise ValueError('New world size cannot preserve global batch')
    if validate_only:
        return {'source_world_size': old_world, 'target_world_size': world_size,
                'global_batch': global_batch, 'step': int(model['global_steps']),
                'group_numels': lengths, 'validation': 'passed'}
    destination.mkdir(parents=True)
    (destination/'pytorch_model').mkdir()
    report = {'source': str(source), 'old_world_size': old_world, 'new_world_size': world_size,
              'step': int(model['global_steps']), 'global_batch': global_batch,
              'group_numels': lengths, 'verified_shards': []}
    for rank in range(world_size):
        out = {k: deepcopy(v) for k,v in template.items() if k not in
               {'base_optimizer_state','single_partition_of_fp32_groups','param_slice_mappings','group_paddings','partition_count'}}
        out.update(base_optimizer_state={'state': {}, 'param_groups': deepcopy(groups)},
                   single_partition_of_fp32_groups=[], param_slice_mappings=[],
                   group_paddings=[], partition_count=[world_size]*len(groups))
        fingerprints = {}
        for i,n in enumerate(lengths):
            size, begin = new_sizes[i], rank*new_sizes[i]
            valid = max(0, min(size,n-begin))
            padding = size-valid
            fp32 = slice_partitions([s['single_partition_of_fp32_groups'][i] for s in shards], begin, valid, old_sizes[i])
            out['single_partition_of_fp32_groups'].append(fp32)
            out['group_paddings'].append(padding)
            out['param_slice_mappings'].append(fragment_map(shapes[i],rank,size,address_type))
            param_id = groups[i]['params'][0]
            original = template['base_optimizer_state']['state'][param_id]
            state = {'step': original['step'].clone()}
            fingerprints[f'{i}/fp32'] = digest_tensor(fp32)
            for key in ['exp_avg','exp_avg_sq']:
                parts = [s['base_optimizer_state']['state'][param_id][key] for s in shards]
                value = torch.zeros(size,dtype=parts[0].dtype)
                value[:valid].copy_(slice_partitions(parts,begin,valid,old_sizes[i]))
                state[key] = value
                fingerprints[f'{i}/{key}'] = digest_tensor(value)
            out['base_optimizer_state']['state'][param_id] = state
        path = destination/f'pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
        torch.save({'optimizer_state_dict': out},path)
        check = torch.load(path,mmap=True,map_location='cpu',weights_only=False)['optimizer_state_dict']
        for i in range(len(groups)):
            if digest_tensor(check['single_partition_of_fp32_groups'][i]) != fingerprints[f'{i}/fp32']:
                raise ValueError(f'FP32 checksum mismatch after writing rank {rank}')
            for key in ['exp_avg','exp_avg_sq']:
                if digest_tensor(check['base_optimizer_state']['state'][groups[i]['params'][0]][key]) != fingerprints[f'{i}/{key}']:
                    raise ValueError(f'{key} checksum mismatch after writing rank {rank}')
        report['verified_shards'].append({'rank':rank,'tensor_sha256':fingerprints,'bytes':path.stat().st_size})
        del out, check
        print(f'Repartitioned and verified rank {rank+1}/{world_size}',flush=True)
    model['dp_world_size'] = world_size
    model['ds_config'] = deepcopy(config)
    model['ds_config']['train_micro_batch_size_per_gpu'] = global_batch//(world_size*accum)
    torch.save(model,destination/'pytorch_model/mp_rank_00_model_states.pt')
    # All ranks load valid Accelerate files; trainer explicitly reseeds after a
    # topology change, since stochastic streams cannot be continued bitwise.
    for rank in range(world_size):
        rng = source/f'random_states_{rank%old_world}.pkl'
        if rng.exists():shutil.copy2(rng,destination/f'random_states_{rank}.pkl')
    for name in ['scheduler.bin','latest','zero_to_fp32.py']:
        if (source/name).exists():shutil.copy2(source/name,destination/name)
    if (source/'trainer_state.json').exists():
        meta=json.loads((source/'trainer_state.json').read_text())
        if meta['global_step'] != model['global_steps']:raise ValueError('Trainer/model steps differ')
        meta.update(resume_world_size=world_size,resume_global_batch_size=global_batch,
                    resume_rng_seed=int(seed)+int(model['global_steps'])*world_size)
        (destination/'trainer_state.json').write_text(json.dumps(meta,indent=2)+'\n')
    after = {str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in source.rglob('*') if p.is_file()}
    if before != after:raise RuntimeError('Source changed during conversion; discard destination')
    (destination/'reshard_complete.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
