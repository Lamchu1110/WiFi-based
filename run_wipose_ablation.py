"""WiPose v4: isolate DSK and temporal trainability; preserve the original baseline."""
import argparse
import csv
import gc
from importlib.metadata import version, PackageNotFoundError
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone

import numpy as np
import torch

from run_wipose_paper import settings as paper_settings
from training import fit
from utils.experiment_archive import record_source
from utils.runtime import RunLock, Tee, atomic_json, replace_with_retry
from wipose.ablation_config import VARIANTS, DEFAULT_VARIANTS, COMPARISONS
from wipose.ablation_data import ablation_bundle
from dataset_lib.training_data import digest


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path)
    p.add_argument('--mode',choices=['preflight','pilot','train'],default='pilot')
    p.add_argument('--stage',choices=['develop','final'],default='develop',
                   help='develop: select on inner validation, no test evaluation; final: full 70%% train, final-epoch test')
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=DEFAULT_VARIANTS)
    p.add_argument('--seeds',nargs='+',type=int,default=[0])
    p.add_argument('--split-seed',type=int,default=41)
    p.add_argument('--split-file',type=Path,help='Optional previously saved full outer 70/30 split.json')
    p.add_argument('--validation-seed',type=int,default=17)
    p.add_argument('--validation-fraction',type=float,default=.1,help='Fraction of the outer training set, develop stage only')
    p.add_argument('--epochs',type=int,help='Default pilot=1, develop train=10, final train=50')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--eval-batch-size',type=int,default=64)
    p.add_argument('--optimizer',choices=['sgd','adamw'],default='sgd')
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--momentum',type=float,default=.9)
    p.add_argument('--weight-decay',type=float,default=0.)
    p.add_argument('--schedule',choices=['constant','cosine'],default='constant')
    p.add_argument('--grad-clip',type=float,default=0.)
    p.add_argument('--device',default=None)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--cache',choices=['mmap','off'],default='mmap')
    p.add_argument('--cache-dir',type=Path,default=Path(__file__).resolve().parent/'data/wipose_cache_v3')
    p.add_argument('--joint-vector-layout',choices=['coordinate-major','joint-major'])
    p.add_argument('--pilot-train-samples',type=int,default=256)
    p.add_argument('--pilot-eval-samples',type=int,default=128)
    p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--parameter-check-every',type=int,default=100,
                   help='Check parameter finiteness every N steps and always before checkpointing; loss/gradients checked each step')
    p.add_argument('--trainability-every',type=int,default=100,
                   help='Sample TCN gradients/actual updates every N steps, first five and epoch end; 0 disables')
    p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs_wipose_v4')
    p.add_argument('--resume-suite',type=Path)
    return p


def settings(args,variant,seed,resume=None):
    args.seed = seed  # paper_settings only supplies shared, explicit defaults.
    value = paper_settings(args,resume)
    for key in ('batch_size','optimizer','lr','momentum','weight_decay','schedule','grad_clip','parameter_check_every','trainability_every'):
        setattr(value,key,getattr(args,key))
    _,value.dsk_layout,value.pooling_mode,value.temporal_block = VARIANTS[variant]
    value.evaluate_test = args.stage == 'final'
    value.selection_policy = 'final' if value.evaluate_test else 'validation'
    value.select_metric = 'MPJPE_xy_px'
    value.select_mode = 'min'
    value.save_predictions = True
    value.source_fingerprint = getattr(args,'source_fingerprint',None)
    return value


def protocol(args):
    return {'version':4,'stage':args.stage,'mode':args.mode,
        'outer_split':'70/30 with seed 41 by default, or imported exact IDs',
        'inner_split':f'{args.validation_fraction:.1%} of outer training set' if args.stage=='develop' else None,
        'selection':'best inner-validation MPJPE; test never evaluated' if args.stage=='develop' else 'fixed final epoch; test once per run',
        'objective':'plain mean MSE on 18 XY joints; confidence is not Z',
        'coordinates':'upstream XY*0.001 during training; metrics in pixels',
        'metrics':'all 18 joints; torso-normalized PCK; PA independently per pose',
        'architecture_controls':{name:dict(zip(('scenario','dsk_layout','pooling_mode','temporal_block'),VARIANTS[name])) for name in args.variants},
        'valid_pairwise_comparisons':[x for x in COMPARISONS if x[0] in args.variants and x[1] in args.variants],
        'initialization':'same backbone initialization for a given seed; short_gated alpha=0 or short_residual zero output projection matches the no-TCN model with the SAME DSK layout and pooling, not necessarily original HPE-Li',
        'trainability_log':'sampled actual train gradients, parameter updates and residual/input; append-only attempt IDs on resume',
        'loss_extensions':'none in this experiment; isolates architecture',
        'precision':args.precision,'tf32':False,
        'limits':['random-frame evaluation, not cross-subject or ordered pose sequences',
                  'five CSI samples per image frame, not five successive pose frames',
                  'corrected DSK is a new implementation, not an exact published result',
                  'pilot is a smoke test; neither pilot success nor shape tests imply improved accuracy']}


METRICS = ('MPJPE_xy_px','PA_MPJPE_xy_px','PCK20','PCK50','R.Wrist_PCK20','L.Wrist_PCK20','R.Wrist_PCK50','L.Wrist_PCK50')


def compare(suite):
    saved = json.loads((suite/'suite_config.json').read_text(encoding='utf-8'))
    rows,contracts = [],[]
    evaluation = 'test' if saved['stage']=='final' else 'validation'
    for seed in saved['seeds']:
        for variant in saved['variants']:
            out = suite/f'seed_{seed}'/variant
            if not (out/'status.json').exists() or json.loads((out/'status.json').read_text())['status']!='complete':
                continue
            config=json.loads((out/'config.json').read_text(encoding='utf-8'))
            contract=config['contract']
            architecture=tuple(contract[key] for key in ('scenario','dsk_layout','pooling_mode','temporal_block'))
            if architecture != VARIANTS[variant] or contract['seed']!=seed:
                raise ValueError('Run architecture/seed does not match the registered variant')
            contracts.append({key:value for key,value in contract.items() if key not in {
                'scenario','seed','dsk_layout','pooling_mode','temporal_block'}})
            metric=json.loads((out/f'{evaluation}_metrics.json').read_text())
            rows.append({'variant':variant,'seed':seed,'evaluation_split':evaluation,
                'selected_epoch':metric.get('evaluated_epoch',metric.get('best_epoch')),
                'epochs':contract['epochs'],'coordinate_unit':'px',**{key:metric[key] for key in METRICS},
                **{key:config[key] for key in ('train_samples','val_samples','test_samples')}})
    if contracts and any(contract!=contracts[0] for contract in contracts):
        raise ValueError('Cannot compare different training/data/evaluation/source contracts')
    aggregated=[]
    for variant in saved['variants']:
        selected=[row for row in rows if row['variant']==variant]
        if not selected:
            continue
        summary={'variant':variant,'runs':len(selected),'seeds':[row['seed'] for row in selected]}
        for key in METRICS:
            values=[row[key] for row in selected if row[key] is not None]
            summary[key+'_mean']=float(np.mean(values)) if values else None
            summary[key+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else None
        aggregated.append(summary)
    differences=[]
    for before,after,label in COMPARISONS:
        for seed in saved['seeds']:
            pair=[next((row for row in rows if row['variant']==name and row['seed']==seed),None) for name in (before,after)]
            if all(pair):
                differences.append({'before':before,'after':after,'seed':seed,'change':label,
                    **{key:pair[1][key]-pair[0][key] if pair[0][key] is not None and pair[1][key] is not None else None for key in METRICS}})
    result={'status':'complete' if len(rows)==len(saved['seeds'])*len(saved['variants']) else 'partial',
            'evaluation_split':evaluation,'rows':rows,'mean_and_sample_std':aggregated,'paired_differences':differences}
    atomic_json(suite/'comparison.json',result)
    if rows:
        temporary=suite/'comparison.csv.tmp'
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        replace_with_retry(temporary,suite/'comparison.csv')
        print(f'COMPARISON ({evaluation}; pixel errors, PCK in %)',flush=True)
        for row in rows:
            print(f'{row["variant"]:20} seed={row["seed"]} epoch={row["selected_epoch"]} '+
                  ' '.join(f'{key}={row[key]:.4f}' if row[key] is not None else f'{key}=n/a' for key in METRICS[:4]),flush=True)
    return result


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=make_parser()
    argv=list(sys.argv[1:] if argv is None else argv)
    args=parser.parse_args(argv)
    resume=bool(args.resume_suite)
    if resume:
        if any(a.startswith('--') and a.split('=')[0] not in ('--resume-suite','--device','--workers') for a in argv):
            parser.error('Resume restores all saved settings; only device/workers can be overridden')
        suite=args.resume_suite.resolve()
        device=args.device
        workers=args.workers if any(a.split('=')[0]=='--workers' for a in argv) else None
        saved=json.loads((suite/'suite_config.json').read_text(encoding='utf-8'))
        if saved.pop('ablation_suite_version',None)!=4:
            parser.error('This is not a v4 ablation suite; use its original branch to resume an old experiment')
        args=argparse.Namespace(**saved)
        args.resume_suite=suite
        if device is not None:args.device=device
        if workers is not None:args.workers=workers
        args.max_steps=0
    else:
        if not args.dataset_root:parser.error('--dataset-root is required')
        args.device=args.device or 'cuda'
        args.epochs=args.epochs if args.epochs is not None else (1 if args.mode=='pilot' else 10 if args.stage=='develop' else 50)
        stamp=datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
        suite=args.output.resolve()/f'{args.mode}_{args.stage}_{stamp}'
    for field in ('dataset_root','cache_dir','output','split_file'):
        value=getattr(args,field)
        if value is not None:setattr(args,field,Path(value).resolve())
    if (min(args.epochs,args.batch_size,args.eval_batch_size,args.pilot_train_samples,args.pilot_eval_samples,args.checkpoint_every,args.parameter_check_every)<1
            or args.batch_size < 2
            or min(args.workers,args.max_steps,args.split_seed,args.validation_seed,args.trainability_every,*args.seeds)<0
            or len(set(args.seeds))!=len(args.seeds) or len(set(args.variants))!=len(args.variants)):
        parser.error('Invalid counts, seeds, or duplicate variants')
    numeric=(args.lr,args.weight_decay,args.momentum,args.grad_clip,args.validation_fraction)
    if not all(math.isfinite(x) for x in numeric) or args.lr<=0 or min(args.weight_decay,args.grad_clip,args.momentum)<0 or not 0<args.validation_fraction<1:
        parser.error('Invalid learning rate, weights or validation fraction')
    if args.deterministic:os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    suite.mkdir(parents=True,exist_ok=resume)
    with RunLock(suite),(suite/'suite.log').open('a',encoding='utf-8') as stream:
        with redirect_stdout(Tee(sys.stdout,stream)),redirect_stderr(Tee(sys.stderr,stream)):
            print(f'Suite: {suite}',flush=True)
            try:
                if not resume:
                    atomic_json(suite/'suite_config.json',{'ablation_suite_version':4,**vars(args)})
                    atomic_json(suite/'protocol.json',protocol(args))
                    atomic_json(suite/'command.json',{'argv':argv,'executable':sys.executable,'cwd':str(Path.cwd())})
                args.source_fingerprint=record_source(Path(__file__).resolve().parent,suite,resume)
                if args.device.startswith('cuda') and not torch.cuda.is_available():
                    raise ValueError('CUDA unavailable; activate a CUDA PyTorch environment or use --device cpu')
                packages={}
                for name in ('numpy','scipy','scikit-learn','h5py','mat73','torchvision'):
                    try:packages[name]=version(name)
                    except PackageNotFoundError:packages[name]=None
                environment={'python':sys.version,'torch':str(torch.__version__),'platform':platform.platform(),
                    'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None,
                    'workers':args.workers,'precision':args.precision,'tf32':False,'packages':packages}
                atomic_json(suite/('resume_environment.json' if resume else 'environment.json'),environment)
                print(json.dumps(environment),flush=True)
                bundle=ablation_bundle(args)
                if resume and (suite/'split.json').exists():
                    previous=json.loads((suite/'split.json').read_text(encoding='utf-8'))
                    if digest(previous)!=digest(bundle.split):
                        raise ValueError('Dataset or split changed; start a new suite')
                else:
                    atomic_json(suite/'split.json',bundle.split)
                print(f'{args.stage}: train={len(bundle.train)} val={len(bundle.val) if bundle.val else 0} heldout_test={len(bundle.test)}',flush=True)
                for seed in args.seeds:
                    seed_dir=suite/f'seed_{seed}';seed_dir.mkdir(exist_ok=True)
                    for variant in args.variants:
                        out=seed_dir/variant;out.mkdir(exist_ok=resume)
                        old=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
                        for filename in ('config.json','preflight.json'):
                            path=out/filename
                            if path.exists() and json.loads(path.read_text())['contract']['data_fingerprint']!=bundle.fingerprint:
                                raise ValueError('Dataset or split changed; start a new suite')
                        if old.get('status') in ('complete','preflight_pass'):
                            print(f'{variant} seed={seed}: preserving completed run',flush=True);continue
                        checkpoint=out/'last.pt' if resume and (out/'last.pt').exists() else None
                        with RunLock(out),(out/'train.log').open('a',encoding='utf-8') as log:
                            with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                                atomic_json(out/'status.json',{'status':'running'})
                                try:
                                    status=fit(settings(args,variant,seed,checkpoint),'wipose',VARIANTS[variant][0],bundle,out)
                                except Exception:
                                    atomic_json(out/'status.json',{'status':'failed','error':traceback.format_exc()});raise
                                atomic_json(out/'status.json',{'status':status})
                        if status=='interrupted':
                            atomic_json(suite/'status.json',{'status':'interrupted'})
                            print(f'Resume: python run_wipose_ablation.py --resume-suite "{suite}"',flush=True)
                            return 130
                        if args.mode!='preflight':compare(suite)
                        gc.collect()
                        if torch.cuda.is_available():torch.cuda.empty_cache()
                status='preflight_pass' if args.mode=='preflight' else compare(suite)['status']
                atomic_json(suite/'status.json',{'status':status})
                print(f'Suite {status}: {suite}',flush=True)
                return 0
            except KeyboardInterrupt:
                atomic_json(suite/'status.json',{'status':'interrupted'})
                print(f'Resume: python run_wipose_ablation.py --resume-suite "{suite}"',flush=True)
                return 130
            except Exception:
                traceback.print_exc()
                atomic_json(suite/('resume_error.json' if resume else 'status.json'),{'status':'failed','error':traceback.format_exc()})
                return 1


if __name__=='__main__':
    raise SystemExit(main())
