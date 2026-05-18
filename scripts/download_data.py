from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import kagglehub
import pandas as pd
from datasets import load_dataset

from configs.experiment_settings import (
    DATASET_VERSION_CONFIG,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
KAGGLE_DOWNLOAD_DIR = RAW_DATA_DIR / "_kaggle_download"

AG_TRAIN_OUTPUT_PATH = RAW_DATA_DIR / "AG_train.csv"
AG_TEST_OUTPUT_PATH = RAW_DATA_DIR / "AG_test.csv"
KAGGLE_NEWS_OUTPUT_PATH = RAW_DATA_DIR / "Kaggle_News.json"
RAW_DATA_METADATA_PATH = RAW_DATA_DIR / "raw_data_metadata.json"

KAGGLE_NEWS_SOURCE_FILENAME = "News_Category_Dataset_v3.json"


def ensure_raw_data_directory_exists() -> None:
    """Create the raw-data directory when it does not yet exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_raw_data_metadata() -> None:
    """Save configured source and revision information for raw datasets."""
    metadata: dict[str, Any] = {
        "ag_news_source": DATASET_VERSION_CONFIG["ag_news_source"],
        "ag_news_revision": DATASET_VERSION_CONFIG["ag_news_revision"],
        "kaggle_news_source": DATASET_VERSION_CONFIG[
            "kaggle_news_source"
        ],
        "kaggle_news_revision": DATASET_VERSION_CONFIG[
            "kaggle_news_revision"
        ],
        "ag_train_output_path": str(AG_TRAIN_OUTPUT_PATH),
        "ag_test_output_path": str(AG_TEST_OUTPUT_PATH),
        "kaggle_news_output_path": str(KAGGLE_NEWS_OUTPUT_PATH),
    }

    with open(
        RAW_DATA_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)


def download_ag_news() -> None:
    """Download AG News and save its train and test splits locally."""
    if AG_TRAIN_OUTPUT_PATH.exists() and AG_TEST_OUTPUT_PATH.exists():
        print("AG News files already exist. Skipping AG News download.")
        return

    load_dataset_kwargs: dict[str, Any] = {}

    if DATASET_VERSION_CONFIG["ag_news_revision"] is not None:
        load_dataset_kwargs["revision"] = DATASET_VERSION_CONFIG[
            "ag_news_revision"
        ]

    dataset = load_dataset(
        "fancyzhx/ag_news",
        **load_dataset_kwargs,
    )

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

    print("AG News download completed.")
    print(f"Training file saved to: {AG_TRAIN_OUTPUT_PATH}")
    print(f"Test file saved to: {AG_TEST_OUTPUT_PATH}")


def find_downloaded_kaggle_news_file(download_directory: Path) -> Path:
    """Locate the downloaded Kaggle News JSON file in the temporary folder."""
    matching_files = list(download_directory.rglob(KAGGLE_NEWS_SOURCE_FILENAME))

    if not matching_files:
        raise FileNotFoundError(
            f"Could not find {KAGGLE_NEWS_SOURCE_FILENAME} "
            f"inside {download_directory}."
        )

    if len(matching_files) > 1:
        raise RuntimeError(
            f"Multiple files named {KAGGLE_NEWS_SOURCE_FILENAME} were found: "
            f"{[str(path) for path in matching_files]}"
        )

    return matching_files[0]


def prepare_kaggle_download_directory() -> None:
    """Create a clean temporary directory for the Kaggle download."""
    if KAGGLE_DOWNLOAD_DIR.exists():
        shutil.rmtree(KAGGLE_DOWNLOAD_DIR)

    KAGGLE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def download_kaggle_news_category_dataset() -> None:
    """Download the Kaggle News Category dataset and save its raw JSON file."""
    if KAGGLE_NEWS_OUTPUT_PATH.exists():
        print("Kaggle News file already exists. Skipping Kaggle download.")
        return

    prepare_kaggle_download_directory()

    downloaded_path = kagglehub.dataset_download(
        "rmisra/news-category-dataset",
        output_dir=str(KAGGLE_DOWNLOAD_DIR),
    )

    downloaded_directory = Path(downloaded_path)

    source_file_path = find_downloaded_kaggle_news_file(downloaded_directory)

    shutil.copy2(
        source_file_path,
        KAGGLE_NEWS_OUTPUT_PATH,
    )

    shutil.rmtree(KAGGLE_DOWNLOAD_DIR, ignore_errors=True)

    print("Kaggle News Category Dataset download completed.")
    print(f"Raw JSON file saved to: {KAGGLE_NEWS_OUTPUT_PATH}")


def verify_required_raw_files_exist() -> None:
    """Confirm that all raw files required by preprocessing exist."""
    required_paths = [
        AG_TRAIN_OUTPUT_PATH,
        AG_TEST_OUTPUT_PATH,
        KAGGLE_NEWS_OUTPUT_PATH,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Raw data setup did not complete successfully. "
            "The following files are missing:\n"
            + "\n".join(missing_paths)
        )


def print_completion_message() -> None:
    """Print a short completion summary after raw-data setup."""
    print()
    print("Raw data setup completed successfully.")
    print("The following files are ready:")
    print(f"- {AG_TRAIN_OUTPUT_PATH}")
    print(f"- {AG_TEST_OUTPUT_PATH}")
    print(f"- {KAGGLE_NEWS_OUTPUT_PATH}")
    print(f"- {RAW_DATA_METADATA_PATH}")


def main() -> None:
    """Download raw datasets, verify outputs, and save source metadata."""
    ensure_raw_data_directory_exists()
    download_ag_news()
    download_kaggle_news_category_dataset()
    verify_required_raw_files_exist()
    save_raw_data_metadata()
    print_completion_message()


if __name__ == "__main__":
    main()
