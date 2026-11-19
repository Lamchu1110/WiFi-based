"""Run a controlled XY WiPose baseline/CSI-TCN pair, with resumable suites."""
import argparse
import csv
import gc
import json
import platform
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Subset

from dataset_lib.training_data import Bundle, digest, wipose_bundle
from training import fit
from utils.runtime import RunLock, Tee, atomic_json


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,help='Extracted directory containing Train/ and Test/')
    parser.add_argument('--mode',choices=['pilot','train','preflight'],default='pilot')
    parser.add_argument('--epochs',type=int,help='Default: pilot=1; train=50. Start a new suite to change this.')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--eval-batch-size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default=None,help='Default: cuda; CPU is available for smoke tests')
    parser.add_argument('--objective',choices=['upstream','normalized'],default='upstream',
                        help='Both models share this objective. Upstream: MSE(conf*pred,conf*target)/32.')
    parser.add_argument('--joint-vector-layout',choices=['coordinate-major','joint-major'])
    parser.add_argument('--wipose-coordinate-unit',choices=['pixel','mm'],default='pixel',
                        help='Native image keypoints are pixels. Use mm only for verified physical coordinates.')
    parser.add_argument('--pilot-train-samples',type=int,default=256)
    parser.add_argument('--pilot-eval-samples',type=int,default=128)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs_wipose_trial')
    parser.add_argument('--resume-suite',type=Path,help='Resume both runs using the saved suite_config.json')
    return parser


def prepare_bundle(args):
    bundle = wipose_bundle(SimpleNamespace(dataset_root=args.dataset_root,npz_dir=None,
                          joint_vector_layout=args.joint_vector_layout,
                          wipose_coordinate_unit=args.wipose_coordinate_unit))
    if args.mode == 'pilot':
        sets,selection = [],{}
        for number,(name,dataset,limit) in enumerate(zip(('train','val','test'),
                (bundle.train,bundle.val,bundle.test),
                (args.pilot_train_samples,args.pilot_eval_samples,args.pilot_eval_samples))):
            ids = sorted(np.random.default_rng(args.seed+number).choice(len(dataset),min(limit,len(dataset)),replace=False).tolist())
            sets.append(Subset(dataset,ids))
            selection[name+'_indices_in_full_split'] = ids
        split = {**bundle.split,'pilot':selection}
        bundle = Bundle(*sets,bundle.target_format,digest([bundle.fingerprint,selection]),split,1,bundle.coordinate_unit)
    return bundle


def training_settings(args, checkpoint=None):
    return SimpleNamespace(epochs=args.epochs,batch_size=args.batch_size,eval_batch_size=args.eval_batch_size,
        lr=args.lr,momentum=0.9,weight_decay=0.01,optimizer='adamw',schedule='upstream_wipose',
        keep_last_batch=True,seed=args.seed,device=args.device,amp=False,amp_dtype='bfloat16',
        temporal_axis='csi',legacy_objective=args.objective=='upstream',grad_clip=0.0,
        warmup_epochs=10,lambda_bone=0.1,lambda_smooth=0.05,lambda_sym=0.02,symmetry_mode='target',
        deterministic=False,select_metric='PCK20',select_mode='max',checkpoint_every=200,
        log_every=10 if args.mode=='pilot' else 100,max_steps=0,
        preflight_only=args.mode=='preflight',resume=checkpoint)


def write_comparison(suite):
    rows=[]
    contracts=[]
    for name in ('baseline','tcn'):
        out=suite/name
        status=json.loads((out/'status.json').read_text(encoding='utf-8')) if (out/'status.json').exists() else {}
        metrics_path=out/'test_metrics.json'
        if status.get('status')!='complete' or not metrics_path.exists():
            continue
        config=json.loads((out/'config.json').read_text(encoding='utf-8'))
        contracts.append(config['contract'])
        metrics=json.loads(metrics_path.read_text(encoding='utf-8'))
        unit='px' if config['contract']['coordinate_unit']=='scaled_pixel' else 'mm'
        row={'scenario':name,'best_epoch':metrics['best_epoch'],'epochs':config['contract']['epochs'],
             'coordinate_unit':unit,'MPJPE':metrics[f'MPJPE_xy_{unit}'],
             'PA_MPJPE':metrics[f'PA_MPJPE_xy_{unit}'],
             **{k:metrics[k] for k in ('PCK20','PCK50','R.Wrist_PCK20','L.Wrist_PCK20','R.Wrist_PCK50','L.Wrist_PCK50')},
             'train_samples':config['train_samples'],'val_samples':config['val_samples'],'test_samples':config['test_samples']}
        rows.append(row)
    if len(contracts)==2:
        a,b=({k:v for k,v in c.items() if k!='scenario'} for c in contracts)
        if a!=b:
            raise ValueError('Refusing to compare runs with different training/data contracts')
    result={'status':'complete' if len(rows)==2 else 'partial','rows':rows,
            'comparison':'HPEWiPoseModel backbone vs CSI-TCN with preserved T=5; both XY, same data/objective/recipe',
            'interpretation':'PCK is percent at 0.2/0.5 torso length; MPJPE units are declared. Pilot subset metrics are not benchmark results.'}
    if len(rows)==2:
        keys=('MPJPE','PA_MPJPE','PCK20','PCK50','R.Wrist_PCK20','L.Wrist_PCK20','R.Wrist_PCK50','L.Wrist_PCK50')
        result['tcn_minus_baseline']={k:rows[1][k]-rows[0][k] if rows[1][k] is not None and rows[0][k] is not None else None for k in keys}
    atomic_json(suite/'comparison.json',result)
    if rows:
        temporary=suite/'comparison.csv.tmp'
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        temporary.replace(suite/'comparison.csv')
        print('\nScenario       MPJPE       PA-MPJPE      PCK20      PCK50    Unit',flush=True)
        for row in rows:
            def value(key):
                return f'{row[key]:.4f}' if row[key] is not None else 'n/a'
            print(f"{row['scenario']:12} {value('MPJPE'):>10} {value('PA_MPJPE'):>14} {value('PCK20'):>10} {value('PCK50'):>10}    {row['coordinate_unit']}",flush=True)
    return result


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=make_parser()
    argv=list(sys.argv[1:] if argv is None else argv)
    args=parser.parse_args(argv)
    if args.resume_suite:
        if any(arg.startswith('--') and arg.split('=',1)[0] not in ('--resume-suite','--device') for arg in argv):
            parser.error('Resume uses suite_config.json; only --resume-suite and optional --device may be supplied')
        suite=args.resume_suite.resolve()
        saved=json.loads((suite/'suite_config.json').read_text(encoding='utf-8'))
        if saved.pop('trial_version',None)!=1:
            parser.error('Unsupported trial suite version')
        override=args.device
        args=argparse.Namespace(**saved)
        for field in ('dataset_root','output'):
            setattr(args,field,Path(getattr(args,field)))
        args.resume_suite=suite
        if override:
            args.device=override
    else:
        if not args.dataset_root:
            parser.error('--dataset-root is required for a new suite')
        args.dataset_root=args.dataset_root.resolve()
        args.device=args.device or 'cuda'
        args.epochs=args.epochs if args.epochs is not None else (1 if args.mode=='pilot' else 50)
        stamp=datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
        suite=args.output.resolve()/f'{args.mode}_{stamp}'
    if args.batch_size<2 or args.eval_batch_size<1 or args.epochs<1 or not np.isfinite(args.lr) or args.lr<=0:
        parser.error('Require epochs>0, batch-size>=2, eval-batch-size>0 and finite positive lr')
    if args.seed<0 or min(args.pilot_train_samples,args.pilot_eval_samples)<1:
        parser.error('Require seed>=0 and positive pilot sample limits')
    suite.mkdir(parents=True,exist_ok=bool(args.resume_suite))
    with RunLock(suite), (suite/'suite.log').open('a',encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
            print(f'Suite: {suite}',flush=True)
            try:
                if not args.resume_suite:
                    atomic_json(suite/'suite_config.json',{'trial_version':1,**vars(args)})
                environment={'python':sys.version,'platform':platform.platform(),'torch':str(torch.__version__),
                             'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
                             'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None}
                atomic_json(suite/('resume_environment.json' if args.resume_suite else 'environment.json'),environment)
                print(json.dumps(environment),flush=True)
                if args.mode=='pilot':
                    print('PILOT: fixed small real-data subsets; results validate the pipeline, not research accuracy.',flush=True)
                print('Building deterministic file split (official Test is split 50/50 for validation/test, seed=41).',flush=True)
                bundle=prepare_bundle(args)
                if args.resume_suite:
                    for scenario in ('baseline','tcn'):
                        for filename in ('config.json','preflight.json'):
                            previous_config=suite/scenario/filename
                            if previous_config.exists():
                                contract=json.loads(previous_config.read_text(encoding='utf-8'))['contract']
                                if contract.get('data_fingerprint')!=bundle.fingerprint:
                                    raise ValueError('Dataset changed since this suite was created; start a new suite')
                # Deterministically sample the actual loader before allocating a GPU model.
                for name,dataset in zip(('train','val','test'),(bundle.train,bundle.val,bundle.test)):
                    if not len(dataset):
                        raise ValueError(f'Empty {name} split')
                    for index in np.linspace(0,len(dataset)-1,min(8,len(dataset)),dtype=int):
                        sample=dataset[int(index)]
                        if sample['input_wifi-csi'].shape!=(9,30,5) or sample['output'].shape!=(18,3):
                            raise ValueError(f'Wrong WiPose shape in {name}')
                print(f'Data checked: train={len(bundle.train)}, val={len(bundle.val)}, test={len(bundle.test)}; units={bundle.coordinate_unit}',flush=True)
                atomic_json(suite/'split.json',bundle.split)
                for scenario in ('baseline','tcn'):
                    out=suite/scenario
                    out.mkdir(exist_ok=bool(args.resume_suite))
                    previous=json.loads((out/'status.json').read_text(encoding='utf-8')) if (out/'status.json').exists() else {}
                    if args.resume_suite and previous.get('status') in ('complete','preflight_pass'):
                        print(f'{scenario}: already complete; preserving its results',flush=True)
                        continue
                    checkpoint=out/'last.pt' if args.resume_suite and (out/'last.pt').exists() else None
                    with RunLock(out), (out/'train.log').open('a',encoding='utf-8') as run_log:
                        with redirect_stdout(Tee(sys.stdout,run_log)),redirect_stderr(Tee(sys.stderr,run_log)):
                            atomic_json(out/'status.json',{'status':'running'})
                            try:
                                status=fit(training_settings(args,checkpoint),'wipose',scenario,bundle,out)
                                atomic_json(out/'status.json',{'status':status})
                            except Exception:
                                atomic_json(out/'status.json',{'status':'failed','error':traceback.format_exc()})
                                raise
                    write_comparison(suite)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if status=='interrupted':
                        atomic_json(suite/'status.json',{'status':'interrupted'})
                        print(f'Resume: python run_wipose_comparison.py --resume-suite "{suite}"',flush=True)
                        return 130
                result=write_comparison(suite)
                status='preflight_pass' if args.mode=='preflight' else result['status']
                atomic_json(suite/'status.json',{'status':status})
                report=suite/'split.json' if args.mode=='preflight' else suite/'comparison.csv'
                print(f'Suite {status}. Results: {report}',flush=True)
                return 0
            except KeyboardInterrupt:
                atomic_json(suite/'status.json',{'status':'interrupted'})
                print(f'Resume: python run_wipose_comparison.py --resume-suite "{suite}"',flush=True)
                return 130
            except Exception:
                traceback.print_exc()
                atomic_json(suite/'status.json',{'status':'failed','error':traceback.format_exc()})
                return 1


if __name__=='__main__':
    raise SystemExit(main())
