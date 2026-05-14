from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

AG_TRAIN_OUTPUT_PATH = RAW_DATA_DIR / "AG_train.csv"
AG_TEST_OUTPUT_PATH = RAW_DATA_DIR / "AG_test.csv"


def ensure_raw_data_directory_exists() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_ag_news() -> None:
    dataset = load_dataset("fancyzhx/ag_news")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    required_columns = {"text", "label"}

    if not required_columns.issubset(train_df.columns):
        raise ValueError(
            f"AG News training split is missing required columns: "
            f"{sorted(required_columns - set(train_df.columns))}"
        )

    if not required_columns.issubset(test_df.columns):
        raise ValueError(
            f"AG News test split is missing required columns: "
            f"{sorted(required_columns - set(test_df.columns))}"
        )

    train_df = train_df[["text", "label"]].copy()
    test_df = test_df[["text", "label"]].copy()

    train_df.to_csv(
        AG_TRAIN_OUTPUT_PATH,
        index=False,
    )

    test_df.to_csv(
        AG_TEST_OUTPUT_PATH,
        index=False,
    )


def print_completion_message() -> None:
    print("AG News download completed.")
    print(f"Training file saved to: {AG_TRAIN_OUTPUT_PATH}")
    print(f"Test file saved to: {AG_TEST_OUTPUT_PATH}")
    print()
    print("The Kaggle News Category Dataset must still be downloaded separately.")
    print("See data/README.md for the Kaggle CLI command and expected filename.")


def main() -> None:
    ensure_raw_data_directory_exists()
    download_ag_news()
    print_completion_message()


if __name__ == "__main__":
    main()
