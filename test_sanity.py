"""Run honest dummy-data checks against the implementations present in this repo.

No dataset, checkpoint, training entrypoint, or real training is loaded.
Returns nonzero if any required implementation/check is unavailable or fails.
THOP uses the same profiler as calculate_model.py and comlexity.py.
"""
import copy
import importlib
import traceback
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
from thop import profile


def run():
    from model import OriginalHPE, HPEWiPoseModel, STHPE
    from utils.losses import KinematicPoseLoss
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch: {torch.__version__}; device: {device}")
    print("THOP counts supported module operations; functional ops are not fully covered.")
    print("FLOPs estimate = 2 * THOP MACs; batch=1, per pose (not per window).")
    rows, failures = [], []
    configs = [("mmfi", (4,3,114,10), 17, OriginalHPE),
               ("wipose", (4,9,30,5), 18, HPEWiPoseModel)]

    def backward_check(loss, model, x):
        assert torch.isfinite(loss), "Nonfinite loss"
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g).item() for g in grads)
        assert x.grad is not None and torch.isfinite(x.grad).all()
        return len(grads)

    for dataset, shape, joints, baseline_class in configs:
        print(f"\n[{dataset}] input={shape}", flush=True)
        counts = {}
        for name in ("Baseline", "ST-HPE"):
            try:
                model = (baseline_class() if name == "Baseline" else
                         STHPE(dataset, use_tcn=True, direct_3d=True)).to(device)
                params = sum(p.numel() for p in model.parameters())
                counts[name] = params
                model.train()
                x = torch.randn(shape, device=device, requires_grad=True)
                pred, _ = model(x)
                assert pred.shape == (4, joints, 2 if name == "Baseline" else 3)
                target = torch.randn_like(pred)
                if name == "ST-HPE":
                    criterion = KinematicPoseLoss(dataset).to(device)
                    loss, components = criterion(pred, target, epoch=10, return_components=True)
                    assert components["L_smooth"].item() == 0
                    initial = criterion(pred, target, epoch=0)
                    halfway = criterion(pred, target, epoch=5)
                    torch.testing.assert_close(initial, components["L_mse"])
                    torch.testing.assert_close(halfway, (initial + loss) / 2)
                    for key, value in components.items():
                        assert torch.isfinite(value)
                        print(f"  frame {key}={value.item():.8f}")
                else:
                    loss = torch.nn.functional.mse_loss(pred, target)
                n = backward_check(loss, model, x)
                print(f"{name} PASS: output={tuple(pred.shape)}, loss={loss.item():.8f}, finite gradients={n}")
                if name == "ST-HPE":
                    model.zero_grad(set_to_none=True)
                    sequence = torch.randn((4,4,*shape[1:]), device=device, requires_grad=True)
                    seq_pred, _ = model(sequence)
                    assert seq_pred.shape == (4,4,joints,3)
                    seq_target = torch.randn_like(seq_pred)
                    seq_loss, parts = criterion(seq_pred, seq_target, epoch=10, return_components=True)
                    for key, value in parts.items():
                        assert torch.isfinite(value)
                        print(f"  sequence {key}={value.item():.8f}")
                    assert parts["L_smooth"] > 0
                    backward_check(seq_loss, model, sequence)
                    print("ST-HPE sequence forward/backward and warm-up PASS")
                    # Isolate every loss term to check its own gradient.
                    for key in parts:
                        probe = seq_pred.detach().clone().requires_grad_(True)
                        _, terms = criterion(probe, seq_target, epoch=10, return_components=True)
                        terms[key].backward()
                        assert probe.grad is not None and torch.isfinite(probe.grad).all()
                        assert torch.count_nonzero(probe.grad) > 0
                    print("All four loss component gradients PASS")
                measured = copy.deepcopy(model).eval()
                with torch.no_grad():
                    macs, _ = profile(measured, inputs=(x[:1].detach(),), verbose=False)
                rows.append((dataset, name, f"{params/1e6:.6f}", f"{macs/1e9:.6f}", f"{2*macs/1e9:.6f}"))
            except Exception:
                failures.append(f"{dataset}/{name}")
                traceback.print_exc()
        if len(counts) == 2:
            delta = (counts["ST-HPE"] - counts["Baseline"]) / 1e6
            print(f"Parameter increase: {delta:.6f}M")
            if not 0.2 <= delta <= 0.4:
                failures.append(f"{dataset}: parameter increase outside 0.2M-0.4M")
        try:
            original = baseline_class().to(device).eval()
            disabled = STHPE(dataset, use_tcn=False, direct_3d=False).to(device).eval()
            disabled.backbone.load_state_dict(original.state_dict())
            with torch.no_grad():
                sample = torch.randn(shape, device=device)
                torch.testing.assert_close(original(sample)[0], disabled(sample)[0])
            print("Disabled upgrades match baseline PASS")
        except Exception:
            failures.append(f"{dataset}: ablation compatibility")
            traceback.print_exc()
    print("\nDataset | Model | Params (M), all registered | THOP MACs (G) | estimated FLOPs (G)")
    for row in rows:
        print(" | ".join(row))
    print("WiPose baseline registers unused skunit4; all-parameter counts include it.")
    print("WiPose real loader returns XY + confidence, not verified XYZ labels.")
    print(f"RESULT: {'FAIL' if failures else 'PASS'}; issues={failures}")
    return int(bool(failures))


class Tee:
    """Write to terminal and persistent UTF-8 log simultaneously."""

    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    log_path = Path(__file__).resolve().parent / "logs" / "sanity.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            started = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
            print(f"\n{'=' * 72}\nRUN START: {started}\nLog: {log_path}", flush=True)
            exit_code = 1
            try:
                exit_code = run()
            except Exception:
                traceback.print_exc()
            finally:
                print(f"RUN END: exit_code={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
