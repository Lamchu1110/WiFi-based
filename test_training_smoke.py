"""Exercise all three training/checkpoint/evaluation paths on synthetic disk data."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io
import yaml


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    root = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="mmfi_smoke_") as temporary:
        work = Path(temporary)
        data = work / "Dataset"
        for subject, count in (("S01", 8), ("S02", 4)):
            action = data / "E01" / subject / "A01"
            csi = action / "wifi-csi"
            csi.mkdir(parents=True)
            np.save(action / "ground_truth.npy", rng.normal(size=(297,17,3)).astype("float32"))
            for index in range(1, 298):
                path = csi / f"frame{index:03d}.mat"
                if index <= count:
                    scipy.io.savemat(path, {"CSIamp": rng.random((3,114,10)).astype("float32")})
                else:
                    path.touch()  # Original loader skips empty frames.
        config = {"data_unit": "frame", "modality": "wifi-csi", "protocol": "protocol3",
                  "split_to_use": "manual_split", "manual_split": {
                      "train_dataset": {"subjects": ["S01"], "actions": ["A01"]},
                      "val_dataset": {"subjects": ["S02"], "actions": ["A01"]}}}
        config_path = work / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        output = root / "runs_smoke"
        subprocess.run([sys.executable, str(root / "run_experiments.py"), "--scenario", "all",
                        "--dataset-root", str(data), "--config", str(config_path),
                        "--epochs", "1", "--batch-size", "4", "--window", "2",
                        "--output", str(output)], check=True)
        suite = max(output.iterdir(), key=lambda p: p.name)
        for scenario in ("baseline", "tcn", "full"):
            directory = suite / scenario
            assert json.loads((directory / "status.json").read_text())["status"] == "complete"
            for name in ("metrics.csv", "test_metrics.json", "best.pt", "last.pt", "split_indices.npz"):
                assert (directory / name).is_file(), name
        print(f"Synthetic training smoke PASS: {suite}")


if __name__ == "__main__":
    main()
