"""WiPose comparison using reported ECCV hyperparameters and disclosed choices."""
import argparse
import csv
import gc
import json
import os
import platform
import sys
import traceback
from contextlib import redirect_stderr,redirect_stdout
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from training import fit
from utils.runtime import RunLock,Tee,atomic_json
from wipose.paper_data import paper_bundle


def make_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path)
    p.add_argument('--mode',choices=['preflight','pilot','train'],default='pilot')
    p.add_argument('--epochs',type=int,help='Default train=50, pilot=1; other budgets are labelled deviations')
    p.add_argument('--seed',type=int,default=0,help='Model/shuffle seed; not the dataset split seed')
    p.add_argument('--split-seed',type=int,default=41,help='Explicit choice; author split seed is unavailable')
    p.add_argument('--device',default=None,help='Default cuda; cpu for smoke tests')
    p.add_argument('--workers',type=int,default=0,help='0 avoids Windows spawn overhead for mmap; optionally measure 2 or 4')
    p.add_argument('--eval-batch-size',type=int,default=64)
    p.add_argument('--cache',choices=['mmap','off'],default='mmap')
    p.add_argument('--cache-dir',type=Path,default=Path(__file__).resolve().parent/'data/wipose_cache')
    p.add_argument('--joint-vector-layout',choices=['coordinate-major','joint-major'])
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32',help='BF16 is an optional numerical variant, not the default comparison')
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--pilot-train-samples',type=int,default=256)
    p.add_argument('--pilot-test-samples',type=int,default=128)
    p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--max-steps',type=int,default=0,help='Interrupt safely for resume tests; not a completed training run')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs_wipose_paper')
    p.add_argument('--resume-suite',type=Path)
    return p


def settings(args,resume=None):
    return SimpleNamespace(epochs=args.epochs,batch_size=32,eval_batch_size=args.eval_batch_size,
        lr=0.001,momentum=0.9,weight_decay=0.0,optimizer='sgd',schedule='constant',
        keep_last_batch=True,seed=args.seed,device=args.device,amp=args.precision=='bf16',amp_dtype='bfloat16',
        temporal_axis='csi',legacy_objective=False,objective_mode='paper_mse',grad_clip=0.0,
        warmup_epochs=0,lambda_bone=0.0,lambda_smooth=0.0,lambda_sym=0.0,symmetry_mode='target',
        deterministic=args.deterministic,select_metric='final_epoch',select_mode='max',selection_policy='final',
        metric_visibility='all',workers=args.workers,checkpoint_every=args.checkpoint_every,
        log_every=10 if args.mode=='pilot' else 100,max_steps=args.max_steps,
        preflight_only=args.mode=='preflight',resume=resume)


def protocol(args):
    return {'reported':{'source':'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04496.pdf',
            'pages':[8,9],'epochs':50,'optimizer':'SGD','momentum':0.9,'batch_size':32,'lr':0.001,
            'train_test_ratio':[0.7,0.3],'objective':'MSE'},
        'implementation_choices':{'split_seed':args.split_seed,'model_seed':args.seed,
            'selection':'final epoch, no validation/test model selection (user selected)',
            'scheduler':'constant LR; paper does not specify decay','weight_decay':0.0,
            'mse_reduction':'mean over samples, joints and XY; no confidence weighting or /32',
            'coordinates':'upstream XY*0.001 for training, pixel metrics; no physical calibration',
            'metrics':'all 18 joints; invalid torso excluded from PCK; PA per pose',
            'initialization':'same inherited HPEWiPoseModel backbone/decoder for both scenarios',
            'architecture':'published repository snapshot, not independently proven identical to ECCV implementation',
            'TCN':'includes pooling change preserving five CSI time samples, not pose frames',
            'precision':args.precision,'tf32':False,'cache':args.cache},
        'deviations':(['pilot subset, not benchmark accuracy'] if args.mode=='pilot' else [])+
                     ([f'epochs={args.epochs}, not 50'] if args.epochs!=50 else [])+
                     (['BF16 numerical variant'] if args.precision!='fp32' else []),
        'limits':['author split IDs/seed and validation policy not published in cited description',
                  'random frame split is not cross-subject or cross-sequence evaluation',
                  'native image labels do not justify reporting physical millimetres',
                  'no guarantee of paper table reproduction or improved accuracy']}


def compare(suite):
    rows=[]
    contracts=[]
    for name in ('baseline','tcn'):
        out=suite/name
        if not (out/'status.json').exists() or json.loads((out/'status.json').read_text())['status']!='complete':
            continue
        config=json.loads((out/'config.json').read_text(encoding='utf-8'))
        contract=config['contract']
        if contract['selection_policy']!='final' or contract['objective_mode']!='paper_mse':
            raise ValueError('Comparison requires final-epoch paper-MSE runs')
        contracts.append({k:v for k,v in contract.items() if k!='scenario'})
        metrics=json.loads((out/'test_metrics.json').read_text())
        rows.append({'scenario':name,'evaluated_epoch':metrics['evaluated_epoch'],'epochs':contract['epochs'],
            'coordinate_unit':'px','MPJPE':metrics['MPJPE_xy_px'],'PA_MPJPE':metrics['PA_MPJPE_xy_px'],
            **{k:metrics[k] for k in ('PCK20','PCK50','R.Wrist_PCK20','L.Wrist_PCK20','R.Wrist_PCK50','L.Wrist_PCK50')},
            **{k:config[k] for k in ('train_samples','val_samples','test_samples')}})
    if len(contracts)==2 and contracts[0]!=contracts[1]:
        raise ValueError('Cannot compare different training/data contracts')
    result={'status':'complete' if len(rows)==2 else 'partial','selection':'fixed final epoch, never test-selected',
            'rows':rows,'note':'Reported-paper hyperparameters with documented choices, not exact reproduction claim'}
    if len(rows)==2:
        result['tcn_minus_baseline']={k:rows[1][k]-rows[0][k] if rows[0][k] is not None and rows[1][k] is not None else None
            for k in ('MPJPE','PA_MPJPE','PCK20','PCK50','R.Wrist_PCK20','L.Wrist_PCK20','R.Wrist_PCK50','L.Wrist_PCK50')}
    atomic_json(suite/'comparison.json',result)
    if rows:
        temporary=suite/'comparison.csv.tmp'
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        temporary.replace(suite/'comparison.csv')
        print('Scenario       MPJPE(px)  PA-MPJPE(px)  PCK20(%)  PCK50(%)',flush=True)
        for row in rows:
            values=['n/a' if row[k] is None else f'{row[k]:.4f}' for k in ('MPJPE','PA_MPJPE','PCK20','PCK50')]
            print(f'{row["scenario"]:12} '+ '  '.join(values),flush=True)
    return result


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    p=make_parser()
    argv=list(sys.argv[1:] if argv is None else argv)
    args=p.parse_args(argv)
    if args.resume_suite:
        allowed=('--resume-suite','--device','--workers')
        if any(a.startswith('--') and a.split('=')[0] not in allowed for a in argv):
            p.error('Resume restores saved settings; only device/workers can be overridden')
        suite=args.resume_suite.resolve()
        override_device=args.device
        override_workers=args.workers if any(a.split('=')[0]=='--workers' for a in argv) else None
        saved=json.loads((suite/'suite_config.json').read_text(encoding='utf-8'))
        if saved.pop('paper_suite_version',None)!=1:
            p.error('Not a paper-protocol suite; old AdamW runs must not be resumed here')
        args=argparse.Namespace(**saved)
        args.resume_suite=suite
        if override_device is not None: args.device=override_device
        if override_workers is not None: args.workers=override_workers
        args.max_steps=0
    else:
        if not args.dataset_root: p.error('--dataset-root is required')
        args.device=args.device or 'cuda'
        args.epochs=args.epochs if args.epochs is not None else (1 if args.mode=='pilot' else 50)
        stamp=datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
        suite=args.output.resolve()/f'{args.mode}_{stamp}'
    for field in ('dataset_root','output','cache_dir'):
        setattr(args,field,Path(getattr(args,field)).resolve())
    if min(args.epochs,args.eval_batch_size,args.pilot_train_samples,args.pilot_test_samples,args.checkpoint_every)<1 or min(args.workers,args.seed,args.split_seed,args.max_steps)<0:
        p.error('Invalid epochs/batch/seed/worker/checkpoint/sample settings')
    if args.deterministic: os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    suite.mkdir(parents=True,exist_ok=bool(args.resume_suite))
    with RunLock(suite),(suite/'suite.log').open('a',encoding='utf-8') as stream:
        with redirect_stdout(Tee(sys.stdout,stream)),redirect_stderr(Tee(sys.stderr,stream)):
            print(f'Suite: {suite}',flush=True)
            try:
                if not args.resume_suite:
                    atomic_json(suite/'suite_config.json',{'paper_suite_version':1,**vars(args)})
                    atomic_json(suite/'protocol.json',protocol(args))
                if args.device.startswith('cuda') and not torch.cuda.is_available():
                    raise ValueError('CUDA unavailable; activate your CUDA PyTorch environment or use --device cpu')
                environment={'python':sys.version,'torch':str(torch.__version__),'platform':platform.platform(),
                    'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    'workers':args.workers,'precision':args.precision,'tf32':False}
                atomic_json(suite/('resume_environment.json' if args.resume_suite else 'environment.json'),environment)
                print(json.dumps(environment),flush=True)
                print('SGDM lr=0.001 momentum=0.9 batch=32; plain MSE; 70/30 random split; final epoch selection.',flush=True)
                bundle=paper_bundle(args)
                for name in ('baseline','tcn'):
                    for filename in ('config.json','preflight.json'):
                        path=suite/name/filename
                        if path.exists() and json.loads(path.read_text(encoding='utf-8'))['contract']['data_fingerprint']!=bundle.fingerprint:
                            raise ValueError('Dataset changed; start a new suite')
                for dataset in (bundle.train,bundle.test):
                    for index in np.linspace(0,len(dataset)-1,min(8,len(dataset)),dtype=int):
                        sample=dataset[int(index)]
                        if sample['input_wifi-csi'].shape!=(9,30,5) or sample['output'].shape!=(18,3):
                            raise ValueError('Invalid sample shape')
                atomic_json(suite/'split.json',bundle.split)
                for name in ('baseline','tcn'):
                    out=suite/name
                    out.mkdir(exist_ok=bool(args.resume_suite))
                    old=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
                    if old.get('status') in ('complete','preflight_pass'):
                        print(f'{name}: already complete; preserving results',flush=True)
                        continue
                    resume=out/'last.pt' if args.resume_suite and (out/'last.pt').exists() else None
                    with RunLock(out),(out/'train.log').open('a',encoding='utf-8') as log:
                        with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                            atomic_json(out/'status.json',{'status':'running'})
                            try:
                                status=fit(settings(args,resume),'wipose',name,bundle,out)
                            except Exception:
                                atomic_json(out/'status.json',{'status':'failed','error':traceback.format_exc()})
                                raise
                            atomic_json(out/'status.json',{'status':status})
                    if status=='interrupted':
                        atomic_json(suite/'status.json',{'status':status})
                        print(f'Resume: python run_wipose_paper.py --resume-suite "{suite}"',flush=True)
                        return 130
                    if args.mode!='preflight': compare(suite)
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                status='preflight_pass' if args.mode=='preflight' else compare(suite)['status']
                atomic_json(suite/'status.json',{'status':status})
                print(f'Suite {status}: {suite}',flush=True)
                return 0
            except KeyboardInterrupt:
                atomic_json(suite/'status.json',{'status':'interrupted'})
                print(f'Resume: python run_wipose_paper.py --resume-suite "{suite}"',flush=True)
                return 130
            except Exception:
                traceback.print_exc()
                atomic_json(suite/'status.json',{'status':'failed','error':traceback.format_exc()})
                return 1


if __name__=='__main__':
    raise SystemExit(main())
