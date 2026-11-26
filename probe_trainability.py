"""Small TRAIN-only SGD probe; never scores or loads held-out tensors.

Requires an existing full split to bind sample IDs to the previous experiment.
This is a software/optimization diagnostic, not a validation or test benchmark.
"""
import argparse
import csv
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch

from model import STHPE
from training import seed_all,pose_objective
from utils.experiment_archive import record_source
from utils.runtime import Tee,atomic_json,atomic_save,rng_state
from utils.trainability import TemporalMonitor
from wipose.ablation_config import VARIANTS,DEFAULT_VARIANTS
from wipose.wipose_dataset import WiPoseDataset


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--split-file',type=Path,required=True)
    parser.add_argument('--allow-partial-dataset',action='store_true')
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=DEFAULT_VARIANTS)
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--samples',type=int,default=256)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'runs_trainability')
    args=parser.parse_args(argv)
    if min(args.steps,args.samples,args.threads)<1 or args.batch_size<2 or args.samples%args.batch_size or args.seed<0:
        parser.error('Positive counts and samples divisible by batch size>=2 required')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    stamp=datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
    out=args.output.resolve()/('probe_'+stamp);out.mkdir(parents=True,exist_ok=False)
    with (out/'probe.log').open('x',encoding='utf-8') as stream,redirect_stdout(Tee(sys.stdout,stream)),redirect_stderr(Tee(sys.stderr,stream)):
        try:
            atomic_json(out/'status.json',{'status':'running','purpose':'train-only optimization diagnostic'})
            atomic_json(out/'config.json',vars(args))
            atomic_json(out/'command.json',{'argv':sys.argv if argv is None else argv,'python':sys.executable,'cwd':str(Path.cwd())})
            record_source(Path(__file__).parent,out)
            atomic_json(out/'environment.json',{'python':sys.version,'torch':str(torch.__version__),'platform':platform.platform(),
                'device':args.device,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if args.device.startswith('cuda') else None})
            split=json.loads(args.split_file.read_text(encoding='utf-8-sig'))
            if split.get('pilot'):raise ValueError('Expected a full split, not a pilot split')
            train=set(split['train']);heldout=set(split.get('test',[]))|set(split.get('val',[]))
            if train&heldout:raise ValueError('Split overlaps train and held-out IDs')
            parts=[WiPoseDataset(args.dataset_root,s) for s in ['Train','Test']]
            local={part.directory.name+'/'+name:(part,i) for part in parts for i,name in enumerate(part.file_list)}
            known=train|heldout
            if not set(local)<=known:raise ValueError('Local IDs not found in supplied split')
            if set(local)!=known and not args.allow_partial_dataset:raise ValueError('Partial dataset requires --allow-partial-dataset')
            available=sorted(train&set(local))
            if len(available)<args.samples:raise ValueError('Insufficient matching TRAIN IDs')
            selected=sorted(np.random.default_rng(41).choice(available,args.samples,replace=False).tolist())
            fingerprints=[];loaded=[]
            for name in selected:
                part,i=local[name];path=part.directory/part.file_list[i]
                fingerprints.append({'id':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                loaded.append(part[i])
            x=torch.stack([s['input_wifi-csi'] for s in loaded]).to(args.device)
            target=torch.stack([s['output'] for s in loaded]).to(args.device)
            atomic_json(out/'sample_ids.json',{'source_split_sha256':hashlib.sha256(args.split_file.read_bytes()).hexdigest(),
                'selected_train_ids':selected,'loaded_heldout_count':0,'available_local_train':len(available),
                'full_train_count':len(train),'input_sha256':fingerprints})
            summaries=[]
            for variant in args.variants:
                path=out/variant;path.mkdir()
                scenario,layout,pooling,block=VARIANTS[variant]
                seed_all(args.seed,True)
                model=STHPE('wipose',use_tcn=scenario=='tcn',direct_3d=False,dsk_layout=layout,pooling_mode=pooling,temporal_block=block).to(args.device)
                optimizer=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=0)
                monitor=TemporalMonitor(model.tcn,path/'trainability.jsonl',16)
                contract={'variant':variant,'architecture_version':model.architecture_version,'scenario':scenario,'dsk_layout':layout,
                    'pooling_mode':pooling,'temporal_block':block,'seed':args.seed,'lr':.001,'momentum':.9,'optimizer':'sgd',
                    'objective':'plain XY MSE','coordinate_unit':'pixel*0.001','data':'selected TRAIN IDs only','steps':args.steps}
                atomic_json(path/'config.json',contract)
                seed_all(args.seed+1,True)
                atomic_save(path/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'step':0,'contract':contract,'rng':rng_state()})
                with (path/'losses.csv').open('x',encoding='utf-8',newline='') as f:
                    csv.writer(f).writerow(['step','train_MSE'])
                model.train();losses=[]
                for step in range(args.steps):
                    batches=args.samples//args.batch_size;epoch=step//batches
                    order=torch.randperm(args.samples,generator=torch.Generator().manual_seed(args.seed+epoch))
                    ids=order[(step%batches)*args.batch_size:(step%batches+1)*args.batch_size].to(args.device)
                    optimizer.zero_grad(set_to_none=True);monitor.begin(step+1,epoch+1,step+1==args.steps)
                    loss=pose_objective(model(x[ids])[0],target[ids],'xy_conf',objective_mode='paper_mse')
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
                    monitor.gradients();optimizer.step();monitor.finish();losses.append(float(loss.detach()))
                    with (path/'losses.csv').open('a',encoding='utf-8',newline='') as f:
                        csv.writer(f).writerow([step+1,losses[-1]])
                    if (step+1)%16==0 or step+1==args.steps:
                        atomic_save(path/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                            'step':step+1,'contract':contract,'rng':rng_state()})
                    if (step+1)%32==0 or step==0:print(f'{variant}: step={step+1}/{args.steps} train_MSE={losses[-1]:.8f}',flush=True)
                monitor.close()
                trace=[json.loads(s) for s in (path/'trainability.jsonl').read_text().splitlines()] if (path/'trainability.jsonl').exists() else []
                summary={'variant':variant,'steps':args.steps,'train_MSE_first':losses[0],'train_MSE_last':losses[-1],
                         'last_sampled_update':trace[-1] if trace else None,'test_evaluated':False,'validation_evaluated':False}
                atomic_json(path/'status.json',{'status':'complete'});summaries.append(summary)
                del model,optimizer,monitor
            atomic_json(out/'summary.json',{'purpose':'optimization probe, not a benchmark','models':summaries,
                'limitation':'Repeated small train subset; cannot establish generalization or improvement over baseline.'})
            atomic_json(out/'status.json',{'status':'complete','heldout_tensors_loaded':0})
            print('Probe complete:',out,flush=True)
            return 0
        except KeyboardInterrupt:
            atomic_json(out/'status.json',{'status':'interrupted','note':'Periodic last.pt and append-only logs retained; this diagnostic does not support resume.'})
            return 130
        except Exception:
            traceback.print_exc();atomic_json(out/'status.json',{'status':'failed','error':traceback.format_exc()})
            return 1


if __name__=='__main__':raise SystemExit(main())
