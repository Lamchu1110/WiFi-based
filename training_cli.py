"""CLI shared by MM-Fi and WiPose with explicit experiment contracts."""
import argparse
import os
import sys
import traceback
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timedelta,timezone
from pathlib import Path

import torch

from dataset_lib.training_data import mmfi_bundle,wipose_bundle
from training import SCENARIOS,fit
from utils.runtime import Tee,RunLock,atomic_json


def main(dataset):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=f'{dataset} training v2; new protocols require new checkpoints')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--dataset-root',type=Path,default=os.environ.get('MMFI_ROOT') if dataset=='mmfi' else None)
    source.add_argument('--dataset-zip',type=Path)
    source.add_argument('--npz-dir',type=Path)
    parser.add_argument('--config',type=Path,default=root/'dataset_lib/config.yaml')
    parser.add_argument('--scenario',choices=[*SCENARIOS,'all','xyz_all'])
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--eval-batch-size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--weight-decay',type=float,default=0.0)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--window',type=int,default=8)
    parser.add_argument('--window-stride',type=int,default=0,help='0 means nonoverlapping stride=window')
    parser.add_argument('--split-mode',choices=['sequence','frame'],default='sequence')
    parser.add_argument('--temporal-axis',choices=['auto','csi','pose'],default='auto')
    parser.add_argument('--legacy-objective',action='store_true',help='Explicit old XY Z/confidence-weighted MSE /32; window=1 only')
    parser.add_argument('--warmup-epochs',type=int,default=10)
    parser.add_argument('--lambda-bone',type=float,default=0.1)
    parser.add_argument('--lambda-smooth',type=float,default=0.05)
    parser.add_argument('--lambda-sym',type=float,default=0.02)
    parser.add_argument('--symmetry-mode',choices=['target','prior'],default='target')
    parser.add_argument('--grad-clip',type=float,default=1.0,help='0 disables clipping, finite gradients still checked')
    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--amp-dtype',choices=['bfloat16','float16'],default='bfloat16')
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--select-metric',default='MPJPE_xy_mm')
    parser.add_argument('--select-mode',choices=['min','max'],default='min')
    parser.add_argument('--checkpoint-every',type=int,default=200)
    parser.add_argument('--log-every',type=int,default=100)
    parser.add_argument('--max-steps',type=int,default=0,help='Pilot stop; checkpoint then resume with 0')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--output',type=Path,default=root/('runs' if dataset=='mmfi' else 'runs_wipose'))
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    import math
    numeric = (args.lr,args.momentum,args.weight_decay,args.grad_clip,args.lambda_bone,args.lambda_smooth,args.lambda_sym)
    if not all(math.isfinite(v) and v >= 0 for v in numeric) or args.lr <= 0:
        parser.error('Learning rate must be positive, weights and optimizer values finite/nonnegative')
    if min(args.epochs,args.batch_size,args.eval_batch_size,args.window,args.checkpoint_every,args.log_every) < 1 or args.batch_size < 2:
        parser.error('Positive epochs/batches/window/intervals and batch-size>=2 required')
    if min(args.max_steps,args.window_stride,args.warmup_epochs,args.seed) < 0:
        parser.error('Seed, warmup, stride and max-steps must be nonnegative')
    if dataset=='mmfi' and args.npz_dir or dataset=='wipose' and args.dataset_zip:
        parser.error('MM-Fi supports folder/ZIP; WiPose supports folder/NPZ')
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if args.resume:
        saved = torch.load(args.resume,map_location='cpu',weights_only=True)
        if saved.get('format_version') != 2 or not all(key in saved for key in ('cursor','optimizer','rng')):
            parser.error('V1 checkpoints cannot resume V2 architecture/objective; start a new run')
        if saved['contract']['dataset'] != dataset:
            parser.error('Checkpoint dataset does not match this entrypoint')
        scenario = saved['contract']['scenario']
        if args.scenario and args.scenario != scenario:
            parser.error('Resume only the scenario in the checkpoint')
        scenarios = [scenario]
        suite = args.resume.resolve().parent.parent
    else:
        scenario = args.scenario or ('all' if dataset=='mmfi' else 'baseline')
        scenarios = ['baseline','tcn','full'] if scenario=='all' else ['xyz','xyz_tcn','xyz_kinematic','full'] if scenario=='xyz_all' else [scenario]
        stamp = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
        suite = args.output/stamp
    if args.legacy_objective and any(SCENARIOS[s][1] for s in scenarios):
        parser.error('Legacy XY objective supports baseline/tcn only, not 3D scenarios')
    if dataset=='mmfi' and args.window<2 and any(SCENARIOS[s][2] for s in scenarios):
        parser.error('Kinematic temporal scenarios need --window>=2')
    bundle = None
    for scenario in scenarios:
        out = args.resume.resolve().parent if args.resume else suite/scenario
        out.mkdir(parents=True,exist_ok=bool(args.resume))
        with RunLock(out):
            with (out/'train.log').open('a',encoding='utf-8') as logfile:
                with redirect_stdout(Tee(sys.stdout,logfile)),redirect_stderr(Tee(sys.stderr,logfile)):
                    print(f'Output: {out}',flush=True)
                    try:
                        atomic_json(out/'status.json',{'status':'running'})
                        if bundle is None:
                            bundle = mmfi_bundle(args) if dataset=='mmfi' else wipose_bundle(args,any(SCENARIOS[s][1] for s in scenarios))
                        status = fit(args,dataset,scenario,bundle,out)
                        atomic_json(out/'status.json',{'status':status})
                    except KeyboardInterrupt:
                        atomic_json(out/'status.json',{'status':'interrupted','phase':'data_or_preflight'})
                        return 130
                    except Exception:
                        traceback.print_exc()
                        atomic_json(out/'status.json',{'status':'failed','error':traceback.format_exc(),
                                                     'resume_candidate':str(out/'last.pt') if (out/'last.pt').exists() else None})
                        return 1
        if status == 'interrupted':
            return 0
    return 0
