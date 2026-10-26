from collections import OrderedDict, namedtuple
import json
import math

import pytest
import torch

from wam.training.zero2_reshard import reshard_checkpoint
from wam.utils.samplers import ResumableEpochSampler

Address = namedtuple('Address', ['numel', 'start'])


def make_checkpoint(root):
    (root/'pytorch_model').mkdir(parents=True)
    shapes=[OrderedDict(a=torch.Size([11]), b=torch.Size([8]))]
    groups=[{'params':[0],'lr':0.0123,'name':'wam'}]
    torch.save({'dp_world_size':2,'mp_world_size':1,'param_shapes':shapes,'global_steps':7,
                'module':{'a':torch.arange(11),'b':torch.arange(8)},
                'ds_config':{'train_batch_size':16,'train_micro_batch_size_per_gpu':8,
                             'gradient_accumulation_steps':1,'zero_optimization':{'stage':2}}},
               root/'pytorch_model/mp_rank_00_model_states.pt')
    for rank in range(2):
        start=rank*10
        valid=10 if rank==0 else 9
        values=torch.arange(start,start+valid,dtype=torch.float32,requires_grad=True)
        moments=torch.zeros(10);moments[:valid]=values+100
        mapping=OrderedDict(a=Address(10,0)) if rank==0 else OrderedDict(a=Address(1,0),b=Address(8,1))
        torch.save({'optimizer_state_dict':{'zero_stage':2,'partition_count':[2],
                    'group_paddings':[10-valid],'single_partition_of_fp32_groups':[values],
                    'param_slice_mappings':[mapping],
                    'base_optimizer_state':{'state':{0:{'step':torch.tensor(7.),'exp_avg':moments,'exp_avg_sq':moments+100}},
                                            'param_groups':groups}}},
                   root/f'pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt')
    torch.save({'last_epoch':7,'_last_lr':[0.0123]},root/'scheduler.bin')
    (root/'trainer_state.json').write_text(json.dumps({'global_step':7,'epoch':2,'batch_in_epoch':3}))


def test_repartition_preserves_fp32_moments_schedule_and_data_position(tmp_path):
    source,dest=tmp_path/'original',tmp_path/'resharded'
    make_checkpoint(source)
    before={p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    report=reshard_checkpoint(source,dest,4)
    assert report['global_batch']==16
    shards=[torch.load(dest/f'pytorch_model/bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt',weights_only=False)['optimizer_state_dict'] for r in range(4)]
    torch.testing.assert_close(torch.cat([s['single_partition_of_fp32_groups'][0] for s in shards]),torch.arange(19,dtype=torch.float32))
    for key,offset in [('exp_avg',100),('exp_avg_sq',200)]:
        merged=torch.cat([s['base_optimizer_state']['state'][0][key] for s in shards])[:19]
        torch.testing.assert_close(merged,torch.arange(19,dtype=torch.float32)+offset)
    assert shards[-1]['group_paddings']==[5]
    assert (dest/'scheduler.bin').read_bytes()==before[next(p for p in before if str(p)=='scheduler.bin')]
    meta=json.loads((dest/'trainer_state.json').read_text())
    assert (meta['global_step'],meta['epoch'],meta['batch_in_epoch'])==(7,2,3)
    old=ResumableEpochSampler(range(1000),42,8,2)
    new=ResumableEpochSampler(range(1000),42,4,4)
    for s in [old,new]:s.set_epoch_offset(2);s.set_resume_batch_offset(3)
    assert list(old)==list(new)
    assert before=={p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):reshard_checkpoint(source,dest,4)


def test_corrupt_optimizer_step_is_rejected_before_output(tmp_path):
    source,dest=tmp_path/'original',tmp_path/'resharded';make_checkpoint(source)
    p=source/'pytorch_model/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt'
    state=torch.load(p,weights_only=False);state['optimizer_state_dict']['base_optimizer_state']['state'][0]['step']=torch.tensor(6.)
    torch.save(state,p)
    with pytest.raises(ValueError,match='Adam step'):
        reshard_checkpoint(source,dest,4)
    assert not dest.exists()
