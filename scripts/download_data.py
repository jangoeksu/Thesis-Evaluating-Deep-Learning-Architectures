from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import pandas as pd
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

AG_TRAIN_OUTPUT_PATH = RAW_DATA_DIR / "AG_train.csv"
AG_TEST_OUTPUT_PATH = RAW_DATA_DIR / "AG_test.csv"
KAGGLE_NEWS_OUTPUT_PATH = RAW_DATA_DIR / "Kaggle_News.json"

KAGGLE_NEWS_SOURCE_FILENAME = "News_Category_Dataset_v3.json"


def ensure_raw_data_directory_exists() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_ag_news() -> None:
    if AG_TRAIN_OUTPUT_PATH.exists() and AG_TEST_OUTPUT_PATH.exists():
        print("AG News files already exist. Skipping AG News download.")
        return

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

    print("AG News download completed.")
    print(f"Training file saved to: {AG_TRAIN_OUTPUT_PATH}")
    print(f"Test file saved to: {AG_TEST_OUTPUT_PATH}")


def find_downloaded_kaggle_news_file(download_directory: Path) -> Path:
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


def download_kaggle_news_category_dataset() -> None:
    if KAGGLE_NEWS_OUTPUT_PATH.exists():
        print("Kaggle News file already exists. Skipping Kaggle download.")
        return

    downloaded_path = kagglehub.dataset_download(
        "rmisra/news-category-dataset",
        output_dir=str(RAW_DATA_DIR),
    )

    downloaded_directory = Path(downloaded_path)

    source_file_path = find_downloaded_kaggle_news_file(downloaded_directory)

    shutil.copy2(
        source_file_path,
        KAGGLE_NEWS_OUTPUT_PATH,
    )

    print("Kaggle News Category Dataset download completed.")
    print(f"Raw JSON file saved to: {KAGGLE_NEWS_OUTPUT_PATH}")


def verify_required_raw_files_exist() -> None:
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
    print()
    print("Raw data setup completed successfully.")
    print("The following files are ready:")
    print(f"- {AG_TRAIN_OUTPUT_PATH}")
    print(f"- {AG_TEST_OUTPUT_PATH}")
    print(f"- {KAGGLE_NEWS_OUTPUT_PATH}")


def main() -> None:
    ensure_raw_data_directory_exists()
    download_ag_news()
    download_kaggle_news_category_dataset()
    verify_required_raw_files_exist()
    print_completion_message()


if __name__ == "__main__":
    main()
