from __future__ import annotations

from scripts.download_data import main as download_raw_data
from src.train import run_full_experiment


def main() -> None:
    download_raw_data()
    run_full_experiment()


if __name__ == "__main__":
    main()
