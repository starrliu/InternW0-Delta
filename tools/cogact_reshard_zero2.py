#!/usr/bin/env python
"""Convert a stopped CogACT ZeRO-2 training checkpoint for a new GPU count."""
import argparse
import json
import torch
from wam.training.zero2_reshard import reshard_checkpoint

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--world-size',type=int,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    result=reshard_checkpoint(args.source,args.destination,args.world_size,seed=args.seed,validate_only=args.validate_only)
    print(json.dumps({k:v for k,v in result.items() if k!='verified_shards'},indent=2))
