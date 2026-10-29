"""Version-2 MM-Fi experiment entrypoint."""
from training_cli import main as training_main

def main():
    return training_main("mmfi")

if __name__ == "__main__":
    raise SystemExit(main())
