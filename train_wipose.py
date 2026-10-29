"""Version-2 WiPose entrypoint; XYZ is never inferred from confidence."""
from dataset_lib.training_data import NpzPoses
from training_cli import main as training_main

def main():
    return training_main("wipose")

if __name__ == "__main__":
    raise SystemExit(main())
