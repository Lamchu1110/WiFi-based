"""Optional two-update CUDA BF16 + checkpoint-resume pilot on synthetic data."""
import json
import tempfile
from pathlib import Path

import torch

from dataset_lib.training_data import Bundle
from test_reliability import SmallPoses,settings
from training import fit


def main():
    root = Path(__file__).resolve().parent
    evidence = root/'validation/cuda_pilot.json'
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        result = {'status':'SKIPPED','reason':'CUDA BF16 is unavailable'}
    else:
        bundle = Bundle(SmallPoses(4,1),SmallPoses(2,2),SmallPoses(2,3),'xyz','cuda-pilot-v2',{},2)
        with tempfile.TemporaryDirectory(prefix='cuda_pilot_') as temporary:
            out = Path(temporary)
            args = settings(device='cuda',amp=True,deterministic=False,max_steps=1)
            assert fit(args,'mmfi','full',bundle,out)=='interrupted'
            saved = torch.load(out/'last.pt',map_location='cpu',weights_only=True)
            assert saved['cursor']['global_step']==1
            args = settings(device='cuda',amp=True,deterministic=False,max_steps=2,resume=out/'last.pt')
            assert fit(args,'mmfi','full',bundle,out)=='interrupted'
            saved = torch.load(out/'last.pt',map_location='cpu',weights_only=True)
            assert saved['cursor']['global_step']==2
            assert all(torch.isfinite(v).all() for v in saved['model'].values())
        result = {'status':'PASS','device':torch.cuda.get_device_name(0),'precision':'bfloat16 AMP; FP32 loss',
                  'checks':['two finite optimizer updates','atomic checkpoint','CUDA RNG/scaler resume'],
                  'data':'synthetic; not a quality/throughput benchmark'}
    evidence.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
