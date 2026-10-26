"""Exercise native DeepSpeed save/load across a DP topology change."""
import argparse,json,os
from pathlib import Path
import torch
import deepspeed

class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.a=torch.nn.Parameter(torch.arange(63,dtype=torch.float32)/100)
  self.b=torch.nn.Parameter(torch.arange(29,dtype=torch.float32)/200)
  self.c=torch.nn.Parameter(torch.arange(19,dtype=torch.float32)/80)
  self.frozen=torch.nn.Parameter(torch.tensor([3.0]),requires_grad=False)
 def forward(self):
  return sum((p.float().square().sum() for p in [self.a,self.b,self.c]))

p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--resume');a=p.parse_args()
world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
m=Tiny();opt=torch.optim.AdamW([{'params':[m.a,m.b],'name':'ab'},{'params':[m.c],'name':'c'}],lr=5e-5,betas=(0.9,0.95),weight_decay=0.01)
config={'train_batch_size':16,'train_micro_batch_size_per_gpu':16//world,'gradient_accumulation_steps':1,'bf16':{'enabled':True},'zero_optimization':{'stage':2,'overlap_comm':False,'contiguous_gradients':False},'zero_allow_untested_optimizer':True,'steps_per_print':10000}
engine,_,_,_=deepspeed.initialize(model=m,optimizer=opt,config=config)
if a.resume:
 loaded,_=engine.load_checkpoint(a.resume,tag='pytorch_model')
 assert loaded and engine.global_steps==3
else:
 for _ in range(3):engine.backward(engine());engine.step()
 engine.save_checkpoint(a.out+'/step3',tag='pytorch_model')
engine.backward(engine());engine.step()
assert engine.global_steps==4
engine.save_checkpoint(a.out+'/step4',tag='pytorch_model')
if int(os.environ['RANK'])==0:print('PROBE_OK',json.dumps({'world':world,'step':engine.global_steps,'resume':a.resume}),flush=True)
torch.distributed.destroy_process_group()
