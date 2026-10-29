"""Reproducible MM-Fi ablations; run --help for settings and dataset override."""
import argparse
import csv
import json
import os
import random
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCENARIOS = {"baseline": (False, False, False), "tcn": (True, False, False),
             "full": (True, True, True)}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("MMFI_ROOT"))
    parser.add_argument("--dataset-zip", type=Path, help="Read MMFi_Dataset.zip directly without extraction")
    parser.add_argument("--config", type=Path, default=ROOT / "dataset_lib/config.yaml")
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def preflight(args):
    if args.dataset_root is None:
        candidates = [ROOT / "Dataset", ROOT / "data/MM-Fi", ROOT.parent / "Dataset"]
        args.dataset_root = next((p for p in candidates if (p / "E01").is_dir()), None)
    if not args.dataset_zip and (args.dataset_root is None or not args.dataset_root.is_dir()):
        raise FileNotFoundError("MM-Fi dataset not found. Supply --dataset-root PATH or MMFI_ROOT. "
                                "Expected PATH/E01/S01/A01/ground_truth.npy and wifi-csi/frame001.mat")
    import numpy as np
    import yaml
    from dataset_lib import make_dataset
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config["data_unit"] != "frame" or config["modality"] != "wifi-csi":
        raise ValueError("Expected frame wifi-csi configuration")
    if args.dataset_zip:
        from dataset_lib.mmfi_zip import make_zip_dataset
        training, heldout = make_zip_dataset(args.dataset_zip, config)
    else:
        training, heldout = make_dataset(str(args.dataset_root), config)
    if len(training) < args.batch_size or len(heldout) < 2:
        raise ValueError(f"Insufficient data: train={len(training)}, heldout={len(heldout)}")
    # Validate every indexed path before launching a multi-hour job.
    for dataset in (training, heldout):
        gt_paths = set()
        for item in dataset.data_list:
            if args.dataset_zip:
                continue  # Archive inventory and every GT validated by ZIP adapter.
            for key in ("gt_path", "wifi-csi_path"):
                if not Path(item[key]).is_file():
                    raise FileNotFoundError(item[key])
            gt_paths.add(item["gt_path"])
        for path in gt_paths:
            gt = np.load(path, mmap_mode="r")
            if gt.shape != (297, 17, 3) or not np.isfinite(gt).all():
                raise ValueError(f"Invalid XYZ label array: {path}, {gt.shape}")
        for index in (0, len(dataset) - 1):
            sample = dataset[index]
            csi = sample["input_wifi-csi"]
            if csi.shape != (3, 114, 10) or not np.isfinite(csi).all():
                raise ValueError(f"Invalid CSI at {index}: {csi.shape}")
    print(f"Data ready: train={len(training)}, heldout={len(heldout)}", flush=True)
    return config, training, heldout


def evaluate(model, loader, device):
    import numpy as np
    import torch
    from utils.eval import calulate_error, compute_pck_pckh
    model.eval()
    sums = np.zeros(10)
    count = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["input_wifi-csi"].float().to(device))[0].cpu().numpy()
            truth = batch["output"].numpy()
            xy, gt = pred[..., :2], truth[..., :2]
            mpjpe, pa = calulate_error(xy, gt)
            p20 = compute_pck_pckh(xy.transpose(0, 2, 1), gt.transpose(0, 2, 1), 0.2)
            p50 = compute_pck_pckh(xy.transpose(0, 2, 1), gt.transpose(0, 2, 1), 0.5)
            xyz = calulate_error(pred, truth) if pred.shape[-1] == 3 else (0, 0)
            values = [mpjpe*1000, pa*1000, p20[17], p50[17], p20[13], p50[13],
                      p20[16], p50[16], xyz[0]*1000, xyz[1]*1000]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite evaluation metrics")
            sums += np.asarray(values) * len(pred)
            count += len(pred)
    if not count:
        raise ValueError("Empty evaluation loader")
    keys = ["MPJPE_xy_mm", "PA_MPJPE_xy_mm", "PCK20", "PCK50", "R.Hand_PCK20",
            "R.Hand_PCK50", "L.Hand_PCK20", "L.Hand_PCK50", "MPJPE_xyz_mm", "PA_MPJPE_xyz_mm"]
    result = dict(zip(keys, (sums/count).tolist()))
    if pred.shape[-1] != 3:
        result["MPJPE_xyz_mm"] = result["PA_MPJPE_xyz_mm"] = None
    return result


def train(args, scenario, config, training, heldout, out):
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset, Subset
    from dataset_lib.mmfi import collate_fn_padd
    from model import OriginalHPE, STHPE
    from utils.losses import KinematicPoseLoss
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    use_tcn, direct_3d, use_kinematic_loss = SCENARIOS[scenario]

    class Windows(Dataset):
        def __init__(self, frames, width):
            self.frames, self.width = frames, width
            self.starts = []
            for i in range(len(frames) - width + 1):
                first, last = frames.data_list[i], frames.data_list[i + width - 1]
                if first["gt_path"] == last["gt_path"] and last["idx"] - first["idx"] == width - 1:
                    self.starts.append(i)

        def __len__(self):
            return len(self.starts)

        def __getitem__(self, index):
            samples = [self.frames[self.starts[index]+i] for i in range(self.width)]
            return {"input_wifi-csi": torch.from_numpy(np.stack([s["input_wifi-csi"] for s in samples])).float(),
                    "output": torch.stack([s["output"] for s in samples]).float()}

    train_data = Windows(training, args.window) if use_kinematic_loss else training
    if len(train_data) < args.batch_size:
        raise ValueError("Not enough training samples/windows")
    val_ids, test_ids = train_test_split(np.arange(len(heldout)), test_size=0.5, random_state=41)
    np.savez(out / "split_indices.npz", val=val_ids, test=test_ids)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        generator=generator, collate_fn=None if use_kinematic_loss else collate_fn_padd)
    val_loader = DataLoader(Subset(heldout, val_ids), batch_size=32, collate_fn=collate_fn_padd)
    test_loader = DataLoader(Subset(heldout, test_ids), batch_size=32, collate_fn=collate_fn_padd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (OriginalHPE() if scenario == "baseline" else
             STHPE("mmfi", use_tcn=use_tcn, direct_3d=direct_3d)).to(device)
    criterion = KinematicPoseLoss("mmfi", warmup_epochs=args.warmup_epochs).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1.0 - max(0, epoch+1-20)/31.0)
    metadata = {**vars(args), "scenario": scenario, "use_tcn": use_tcn, "direct_3d": direct_3d,
                "use_kinematic_loss": use_kinematic_loss, "device": str(device),
                "train_samples": len(train_data), "scheduler": "original: 20 constant + linear decay /31",
                "metrics": "common XY; torso-normalized PCK thresholds .2/.5 in percent; XYZ additional for full",
                "legacy_loss": "baseline/tcn preserve original MSE(Z*predXY,Z*gtXY)/32; Z is not confidence",
                "temporal": "full uses overlapping within-action pose windows; baseline/tcn use frames"}
    (out / "config.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metadata, default=str), flush=True)
    best = -1
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = None
        for epoch in range(args.epochs):
            model.train()
            total = samples = 0
            for step, batch in enumerate(loader):
                x, gt = batch["input_wifi-csi"].float().to(device), batch["output"].float().to(device)
                if not torch.isfinite(x).all() or not torch.isfinite(gt).all():
                    raise ValueError("Nonfinite training input/labels")
                optimizer.zero_grad(set_to_none=True)
                pred = model(x)[0]
                loss = criterion(pred, gt, epoch=epoch) if use_kinematic_loss else (
                    torch.nn.functional.mse_loss(pred*gt[...,2:3], gt[...,:2]*gt[...,2:3])/32)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise ValueError("Nonfinite gradient")
                optimizer.step()
                total += loss.item() * len(x)
                samples += len(x)
                if step % 100 == 0:
                    print(f"epoch={epoch+1} step={step}/{len(loader)} loss={loss.item():.8f}", flush=True)
            result = {"epoch": epoch+1, "lr": optimizer.param_groups[0]["lr"],
                      "train_loss": total/samples, **evaluate(model, val_loader, device)}
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(result))
                writer.writeheader()
            writer.writerow(result)
            file.flush()
            print(json.dumps(result), flush=True)
            scheduler.step()
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "epoch": epoch+1, "metrics": result,
                          "metadata": {key: str(value) if isinstance(value, Path) else value for key, value in metadata.items()}}
            torch.save(checkpoint, out / "last.pt")
            if result["PCK20"] > best:
                best = result["PCK20"]
                torch.save(checkpoint, out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model"])
    result = evaluate(model, test_loader, device)
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("TEST " + json.dumps(result), flush=True)


def main():
    from contextlib import redirect_stdout, redirect_stderr
    from test_sanity import Tee
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = arguments()
    if args.epochs < 1 or args.batch_size < 2 or args.window < 2 or args.epochs > 50:
        raise ValueError("Require 1..50 epochs, batch >=2 and window >=2")
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S_%f")
    suite = args.output / stamp
    suite.mkdir(parents=True)
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        out = suite / scenario
        out.mkdir()
        with (out / "train.log").open("w", encoding="utf-8") as logfile:
            with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
                print(f"Scenario={scenario}; output={out}", flush=True)
                try:
                    config, training, heldout = preflight(args)
                    if not args.preflight_only:
                        train(args, scenario, config, training, heldout, out)
                except Exception:
                    traceback.print_exc()
                    (out / "status.json").write_text(json.dumps({"status": "failed", "error": traceback.format_exc()}), encoding="utf-8")
                    return 1
                (out / "status.json").write_text(json.dumps({"status": "preflight_pass" if args.preflight_only else "complete"}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
