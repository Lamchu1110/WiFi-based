"""Shared version-2 trainer: controlled windows, safe resume and streaming metrics."""
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import STHPE
from utils.losses import KinematicPoseLoss
from utils.metrics import PoseMetrics
from utils.runtime import atomic_json,atomic_save,rng_state,restore_rng
from utils.data_loading import OrderedSampler,make_loader


SCENARIOS = {
    'baseline':(False,False,False), 'tcn':(True,False,False),
    'xyz':(False,True,False), 'xyz_tcn':(True,True,False),
    'xyz_kinematic':(False,True,True), 'full':(True,True,True),
}


def pose_objective(pred,gt,target_format,legacy=False,objective_mode=None):
    pred,gt = pred.float(),gt.float()
    target = gt[...,:pred.shape[-1]]
    if objective_mode == 'paper_mse':
        return torch.nn.functional.mse_loss(pred,target)
    if legacy:
        if pred.shape[-1] != 2:
            raise ValueError('Legacy objective only supports XY')
        return torch.nn.functional.mse_loss(pred*gt[...,2:3],target*gt[...,2:3])/32
    if target_format == 'xy_conf':
        confidence = gt[...,2:3]
        if (confidence < 0).any() or not torch.isfinite(confidence).all():
            raise ValueError('Confidence must be finite and nonnegative')
        denominator = confidence.sum()*pred.shape[-1]
        if denominator <= 0:
            raise ValueError('No visible joints in training batch')
        return ((pred-target).square()*confidence).sum()/denominator
    return torch.nn.functional.mse_loss(pred,target)


def evaluate(model,dataset,device,name,target_format,batch_size,coordinate_unit='m',workers=0,metric_visibility='visible'):
    model.eval()
    metrics = PoseMetrics(name,model.direct_3d,coordinate_unit)
    generator = torch.Generator().manual_seed(123456)
    loader = make_loader(dataset,batch_size,device,workers,generator=generator)
    with torch.inference_mode():
        for batch in loader:
            x = batch['input_wifi-csi'].float().to(device,non_blocking=True)
            gt = batch['output'].float()
            prediction = model(x)[0].float().cpu().numpy()
            metrics.update(prediction,gt.numpy(),gt[...,2].numpy() if target_format=='xy_conf' and metric_visibility=='visible' else None)
    return metrics.result()


def seed_all(seed,deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def fit(args,name,scenario,bundle,out):
    optimizer_name = getattr(args,'optimizer','sgd')
    schedule_name = getattr(args,'schedule','v2')
    keep_last = getattr(args,'keep_last_batch',False)
    selection_policy = getattr(args,'selection_policy','validation')
    objective_mode = getattr(args,'objective_mode',None)
    metric_visibility = getattr(args,'metric_visibility','visible')
    workers = getattr(args,'workers',0)
    if selection_policy not in ('validation','final') or workers < 0:
        raise ValueError('Invalid checkpoint selection policy or workers')
    if selection_policy == 'validation' and bundle.val is None:
        raise ValueError('Validation checkpoint selection needs a validation set')
    if getattr(args,'select_metric',None) is None:
        args.select_metric = 'MPJPE_xy_px' if bundle.coordinate_unit == 'scaled_pixel' else 'MPJPE_xy_mm'
    use_tcn,direct_3d,use_kinematic = SCENARIOS[scenario]
    if direct_3d and bundle.target_format != 'xyz':
        raise ValueError('Direct 3D needs verified XYZ labels; confidence is not Z')
    if use_kinematic and bundle.window < 2:
        raise ValueError('Kinematic temporal training needs ordered W>=2 windows')
    temporal_axis = args.temporal_axis
    if temporal_axis == 'auto':
        temporal_axis = 'pose' if bundle.window > 1 else 'csi'
    if args.legacy_objective and (direct_3d or bundle.window != 1):
        raise ValueError('Legacy XY reproduction requires window=1 and an XY scenario')
    if any(not len(s) for s in (bundle.train,bundle.val,bundle.test) if s is not None) or len(bundle.train) < args.batch_size:
        raise ValueError('Insufficient frames/windows; reduce batch/window or verify split')
    if use_tcn and temporal_axis=='pose' and bundle.window < 2:
        raise ValueError('Pose TCN needs windows; native WiPose can use --temporal-axis csi')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this Python environment; install a CUDA PyTorch build or use --device cpu')
    if keep_last and len(bundle.train) % args.batch_size == 1:
        raise ValueError('Last training batch would have one sample; change batch size or disable --keep-last-batch for BatchNorm')
    if args.amp and device.type != 'cuda':
        raise ValueError('--amp currently supports CUDA only')
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
    if args.amp and amp_dtype == torch.bfloat16:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError('This GPU does not support BF16. Leave AMP off or explicitly pilot --amp-dtype float16.')
    contract = {
        'architecture_version':2,'dataset':name,'scenario':scenario,'data_fingerprint':bundle.fingerprint,
        'target_format':bundle.target_format,'window':bundle.window,'batch_size':args.batch_size,
        'eval_batch_size':args.eval_batch_size,
        'seed':args.seed,'epochs':args.epochs,'lr':args.lr,'momentum':args.momentum,
        'weight_decay':args.weight_decay,'temporal_axis':temporal_axis,'legacy_objective':args.legacy_objective,
        'warmup_epochs':args.warmup_epochs,'lambda_bone':args.lambda_bone,'lambda_smooth':args.lambda_smooth,
        'lambda_sym':args.lambda_sym,'symmetry_mode':args.symmetry_mode,'grad_clip':args.grad_clip,
        'amp':args.amp,'amp_dtype':args.amp_dtype,'deterministic':args.deterministic,'selection':args.select_metric,'selection_mode':args.select_mode,
        'optimizer':optimizer_name,'schedule':schedule_name,'keep_last_batch':keep_last,
        'coordinate_unit':bundle.coordinate_unit,
        'selection_policy':selection_policy,'objective_mode':objective_mode,'metric_visibility':metric_visibility,
    }
    if args.resume:
        checkpoint = torch.load(args.resume,map_location='cpu',weights_only=True)
        if checkpoint.get('format_version') != 2 or checkpoint.get('contract') != contract:
            differences = [key for key in contract if checkpoint.get('contract',{}).get(key) != contract[key]]
            raise ValueError(f'Resume contract mismatch: {differences}. V1 checkpoints require a new experiment.')
    else:
        checkpoint = None
    metadata = {**vars(args),'contract':contract,'train_samples':len(bundle.train),
                'val_samples':len(bundle.val) if bundle.val is not None else 0,'test_samples':len(bundle.test),'frames_per_batch':args.batch_size*bundle.window,
                'eval_window_counting':'per window occurrence; default MM-Fi stride equals width, tails dropped',
                'metrics':'XY common; XYZ additionally; torso-normalized PCK; velocity/acceleration within explicit windows',
                'causality':'noncausal window model; not an online/causal latency benchmark'}
    # Preflight and incompatible resume never overwrite a previous run's config/checkpoint.
    if args.preflight_only:
        for dataset in (bundle.train,bundle.val,bundle.test):
            if dataset is None:
                continue
            for index in (0,len(dataset)-1):
                sample = dataset[index]
                if not all(torch.isfinite(v).all() for v in sample.values()):
                    raise ValueError('Nonfinite preflight sample')
        atomic_json(out/'preflight.json',metadata)
        print(f'Preflight PASS: train={len(bundle.train)} val={metadata["val_samples"]} test={len(bundle.test)} frames/batch={args.batch_size*bundle.window}',flush=True)
        return 'preflight_pass'
    seed_all(args.seed,args.deterministic)
    model = STHPE(name,use_tcn=use_tcn,direct_3d=direct_3d,temporal_axis=temporal_axis).to(device)
    criterion = KinematicPoseLoss(name,args.lambda_bone,args.lambda_smooth,args.lambda_sym,
                                  args.warmup_epochs,args.symmetry_mode).to(device)
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')
    # Versioned schedule: fixed 20-epoch plateau (or total epochs if shorter), then linear to 1%.
    plateau = min(20,args.epochs)
    if schedule_name == 'constant':
        lr_scale = lambda epoch: 1.0
    elif schedule_name == 'upstream_wipose':
        # att_wipose.py: fixed 20 epochs, decay over 31; clamp avoids negative LR beyond epoch 50.
        lr_scale = lambda epoch: max(0.0,1.0-max(0,epoch+1-20)/31.0)
    elif schedule_name == 'v2':
        lr_scale = lambda epoch: 1.0 if epoch < plateau else max(0.01,1-(epoch-plateau+1)*0.99/max(1,args.epochs-plateau))
    else:
        raise ValueError(f'Unsupported schedule: {schedule_name}')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scale)
    scaler = torch.amp.GradScaler('cuda',enabled=args.amp and amp_dtype == torch.float16)
    cursor = {'epoch':0,'next_batch':0,'global_step':0,'loss_sum':0.0,'samples':0,
              'components':{},'best_score':None,'best_epoch':None}
    best_model = None
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        cursor,best_model = checkpoint['cursor'],checkpoint['best_model']
        restore_rng(checkpoint['rng'])
    else:
        # Separate training randomness (e.g. dropout) from architecture initialization.
        seed_all(args.seed+1,args.deterministic)
    atomic_json(out/'config.json',metadata)
    atomic_json(out/'split.json',bundle.split)
    print(json.dumps({'scenario':scenario,'device':str(device),'window':bundle.window,'temporal_axis':temporal_axis,
                      'train_samples':len(bundle.train),'frames_per_batch':args.batch_size*bundle.window,
                      'resume_epoch':cursor['epoch'],'resume_batch':cursor['next_batch']}),flush=True)

    def save():
        atomic_save(out/'last.pt',{'format_version':2,'contract':contract,'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                    'cursor':cursor,'best_model':best_model,'rng':rng_state()})

    # Roll CSV back to the checkpoint if a crash occurred after CSV append but before last.pt replace.
    csv_path = out/'metrics.csv'
    rows = []
    if csv_path.exists() and checkpoint:
        with csv_path.open(encoding='utf-8',newline='') as file:
            rows = [row for row in csv.DictReader(file) if int(row['epoch']) <= cursor['epoch']]
    elif csv_path.exists():
        raise FileExistsError('Refusing to overwrite existing metrics without --resume')
    if rows:
        with csv_path.open('w',encoding='utf-8',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif csv_path.exists():
        csv_path.write_text('',encoding='utf-8')
    save()  # Even interruption before the first optimizer update has a resumable checkpoint.
    in_step = False
    started = time.monotonic()
    sampler = OrderedSampler()
    generator = torch.Generator()
    loader = make_loader(bundle.train,args.batch_size,device,workers,sampler,
                         drop_last=not keep_last,generator=generator)
    try:
        while cursor['epoch'] < args.epochs:
            epoch = cursor['epoch']
            model.train()
            permutation = torch.randperm(len(bundle.train),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
            batch_count = math.ceil(len(permutation)/args.batch_size) if keep_last else len(permutation)//args.batch_size
            if cursor['next_batch'] > batch_count:
                raise ValueError('Invalid checkpoint cursor')
            sampler.indices = permutation[cursor['next_batch']*args.batch_size:batch_count*args.batch_size]
            generator.manual_seed(args.seed+epoch+100000)
            timing = {'epoch':epoch+1,'starting_batch':cursor['next_batch'],'data_seconds':0.0,
                      'train_seconds':0.0,'checkpoint_seconds':0.0,'samples':0}
            previous_end = time.perf_counter()
            for batch in loader:
                batch_start = time.perf_counter()
                timing['data_seconds'] += batch_start-previous_end
                in_step = True
                x,gt = batch['input_wifi-csi'].float().to(device,non_blocking=True),batch['output'].float().to(device,non_blocking=True)
                if not torch.isfinite(x).all() or not torch.isfinite(gt).all():
                    raise ValueError('Nonfinite training sample')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=args.amp,dtype=amp_dtype):
                    prediction = model(x)[0]
                if use_kinematic:
                    loss,parts = criterion(prediction,gt,epoch=epoch,return_components=True)
                else:
                    loss = pose_objective(prediction,gt,bundle.target_format,args.legacy_objective,objective_mode)
                    zero = loss.detach()*0
                    parts = {'L_mse':loss,'L_bone':zero,'L_smooth':zero,'L_sym':zero}
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip if args.grad_clip else math.inf,
                                                      error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
                    raise FloatingPointError('Optimizer produced nonfinite parameters; previous checkpoint preserved')
                size = len(x)
                values = torch.stack([loss.detach(),*[v.detach() for v in parts.values()]]).cpu().tolist()
                cursor['loss_sum'] += values[0]*size
                cursor['samples'] += size
                for key,value in zip(parts,values[1:]):
                    cursor['components'][key] = cursor['components'].get(key,0.0)+value*size
                cursor['next_batch'] += 1
                cursor['global_step'] += 1
                in_step = False
                timing['samples'] += size
                timing['train_seconds'] += time.perf_counter()-batch_start
                if cursor['global_step'] % args.log_every == 0 or cursor['global_step']==1:
                    print(f"epoch={epoch+1}/{args.epochs} batch={cursor['next_batch']}/{batch_count} step={cursor['global_step']} loss={loss.item():.8f} grad_norm={float(norm):.5f} elapsed_s={time.monotonic()-started:.1f}",flush=True)
                if cursor['global_step'] % args.checkpoint_every == 0:
                    checkpoint_start = time.perf_counter()
                    save()
                    timing['checkpoint_seconds'] += time.perf_counter()-checkpoint_start
                if args.max_steps and cursor['global_step'] >= args.max_steps:
                    save()
                    print('Stopped at --max-steps; resume from last.pt with the same contract',flush=True)
                    return 'interrupted'
                previous_end = time.perf_counter()
            eval_start = time.perf_counter()
            metrics = evaluate(model,bundle.val,device,name,bundle.target_format,args.eval_batch_size,bundle.coordinate_unit,workers,metric_visibility) if selection_policy=='validation' else {}
            timing['validation_seconds'] = time.perf_counter()-eval_start
            result = {'epoch':epoch+1,'global_step':cursor['global_step'],'lr':optimizer.param_groups[0]['lr'],
                      'train_loss':cursor['loss_sum']/cursor['samples'],
                      **{key:value/cursor['samples'] for key,value in cursor['components'].items()},**metrics}
            if selection_policy=='validation':
                score = result.get(args.select_metric)
                if score is None or not math.isfinite(score):
                    raise ValueError(f'Checkpoint selection metric unavailable: {args.select_metric}')
                improved = cursor['best_score'] is None or (score < cursor['best_score'] if args.select_mode=='min' else score > cursor['best_score'])
                if improved:
                    cursor['best_score'],cursor['best_epoch'] = score,epoch+1
                    best_model = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            else:
                cursor['best_epoch'] = epoch+1
            exists = csv_path.exists() and csv_path.stat().st_size > 0
            with csv_path.open('a',encoding='utf-8',newline='') as file:
                writer = csv.DictWriter(file,fieldnames=list(result))
                if not exists:
                    writer.writeheader()
                writer.writerow(result)
                file.flush()
                import os
                os.fsync(file.fileno())
            print(json.dumps(result,allow_nan=False),flush=True)
            scheduler.step()
            cursor.update(epoch=epoch+1,next_batch=0,loss_sum=0.0,samples=0,components={})
            checkpoint_start = time.perf_counter()
            save()
            if selection_policy=='validation':
                atomic_save(out/'best.pt',{'format_version':2,'contract':contract,'model':best_model,
                                         'best_epoch':cursor['best_epoch'],'best_score':cursor['best_score']})
            timing['checkpoint_seconds'] += time.perf_counter()-checkpoint_start
            with (out/'performance.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(timing)+'\n')
            print('TIMING '+json.dumps(timing),flush=True)
    except KeyboardInterrupt:
        if not in_step:
            save()
        print('Interrupted. Last atomic checkpoint remains available; an in-flight optimizer step is never saved.',flush=True)
        return 'interrupted'
    if selection_policy=='final':
        best_model = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    if best_model is None:
        raise RuntimeError('No validated model available for test')
    # Restore best from last.pt as well: a crash between last.pt and best.pt cannot lose the best weights.
    model.load_state_dict(best_model)
    selected = {'evaluated_epoch':cursor['epoch'],'selection_policy':'final'} if selection_policy=='final' else {
        'best_epoch':cursor['best_epoch'],'best_score':cursor['best_score']}
    atomic_save(out/('final.pt' if selection_policy=='final' else 'best.pt'),
                {'format_version':2,'contract':contract,'model':best_model,**selected})
    test_start = time.perf_counter()
    metrics = evaluate(model,bundle.test,device,name,bundle.target_format,args.eval_batch_size,bundle.coordinate_unit,workers,metric_visibility)
    atomic_json(out/'test_timing.json',{'seconds':time.perf_counter()-test_start,'samples':len(bundle.test)})
    metric_metadata = selected if selection_policy=='final' else {'best_epoch':cursor['best_epoch']}
    atomic_json(out/'test_metrics.json',{**metric_metadata,**metrics})
    print('TEST '+json.dumps(metrics,allow_nan=False),flush=True)
    return 'complete'
