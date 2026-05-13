from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from configs.experiment_settings import (
    AG_LABEL_MAPPING,
    AG_TEST_PATH,
    AG_TRAIN_PATH,
    EXPERIMENT_CONFIG,
    KAGGLE_CATEGORY_MERGE_MAP,
    KAGGLE_NEWS_PATH,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    SEED,
)


def ensure_directories_exist() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_raw_files_exist() -> None:
    required_files = [
        AG_TRAIN_PATH,
        AG_TEST_PATH,
        KAGGLE_NEWS_PATH,
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "The following required raw data files are missing:\n"
            + "\n".join(missing_files)
        )


def clean_text_value(value) -> str:
    if pd.isna(value):
        return ""

    value = str(value)
    value = value.replace("\n", " ")
    value = value.replace("\r", " ")
    value = value.replace("\t", " ")
    value = " ".join(value.split())

    return value.strip()


def combine_headline_and_description(headline, short_description) -> str:
    headline = clean_text_value(headline)
    short_description = clean_text_value(short_description)

    if not headline:
        return short_description

    if not short_description:
        return headline

    if headline.lower() == short_description.lower():
        return headline

    return f"{headline} {short_description}"


def load_ag_news() -> tuple[pd.DataFrame, pd.DataFrame]:
    ag_train_raw = pd.read_csv(
        AG_TRAIN_PATH,
        usecols=["text", "label"],
        dtype={"text": "string", "label": "int64"},
    )

    ag_test_raw = pd.read_csv(
        AG_TEST_PATH,
        usecols=["text", "label"],
        dtype={"text": "string", "label": "int64"},
    )

    return ag_train_raw, ag_test_raw


def load_kaggle_news() -> pd.DataFrame:
    return pd.read_json(
        KAGGLE_NEWS_PATH,
        lines=True,
    )


def validate_ag_news(
    ag_train_raw: pd.DataFrame,
    ag_test_raw: pd.DataFrame,
) -> None:
    expected_columns = ["text", "label"]

    if list(ag_train_raw.columns) != expected_columns:
        raise ValueError(
            f"Unexpected AG News training columns: "
            f"{ag_train_raw.columns.tolist()}"
        )

    if list(ag_test_raw.columns) != expected_columns:
        raise ValueError(
            f"Unexpected AG News test columns: "
            f"{ag_test_raw.columns.tolist()}"
        )

    expected_labels = set(AG_LABEL_MAPPING.keys())
    train_labels = set(ag_train_raw["label"].unique())
    test_labels = set(ag_test_raw["label"].unique())

    if train_labels != expected_labels:
        raise ValueError(
            f"Unexpected AG News training labels: {sorted(train_labels)}"
        )

    if test_labels != expected_labels:
        raise ValueError(
            f"Unexpected AG News test labels: {sorted(test_labels)}"
        )


def validate_kaggle_news(kaggle_raw: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        "link",
        "headline",
        "category",
        "short_description",
        "authors",
        "date",
    ]

    if list(kaggle_raw.columns) != expected_columns:
        raise ValueError(
            f"Unexpected Kaggle News columns: {kaggle_raw.columns.tolist()}"
        )

    original_class_count = kaggle_raw["category"].nunique()

    if original_class_count != EXPERIMENT_CONFIG["kaggle_original_classes"]:
        raise ValueError(
            f"Unexpected number of original Kaggle classes: "
            f"{original_class_count}"
        )

    merge_check = kaggle_raw.copy()
    merge_check["category"] = merge_check["category"].astype(str)
    merge_check["merged_category"] = merge_check["category"].map(
        KAGGLE_CATEGORY_MERGE_MAP
    )

    unmapped_categories = sorted(
        merge_check.loc[
            merge_check["merged_category"].isna(),
            "category",
        ].unique()
    )

    if unmapped_categories:
        raise ValueError(
            f"Unmapped Kaggle categories found: {unmapped_categories}"
        )

    merged_class_count = merge_check["merged_category"].nunique()
    expected_merged_count = EXPERIMENT_CONFIG[
        "kaggle_expected_merged_classes"
    ]

    if merged_class_count != expected_merged_count:
        raise ValueError(
            f"Unexpected number of merged Kaggle classes: "
            f"{merged_class_count}. Expected: {expected_merged_count}"
        )

    return merge_check


def preprocess_ag_news(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].apply(clean_text_value)
    df = df[df["text"].str.len() > 0]
    df = df[df["label"].isin(AG_LABEL_MAPPING.keys())]
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(AG_LABEL_MAPPING)

    return df.reset_index(drop=True)[["text", "label", "label_name"]]


def preprocess_kaggle_news(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, str]]:
    df = df.copy()
    df["category"] = df["category"].astype(str)
    df["merged_category"] = df["category"].map(KAGGLE_CATEGORY_MERGE_MAP)

    unmapped_categories = sorted(
        df.loc[
            df["merged_category"].isna(),
            "category",
        ].unique()
    )

    if unmapped_categories:
        raise ValueError(
            f"Unmapped Kaggle categories found: {unmapped_categories}"
        )

    df["text"] = df.apply(
        lambda row: combine_headline_and_description(
            row["headline"],
            row["short_description"],
        ),
        axis=1,
    )

    df = df[df["text"].str.len() > 0].copy()

    merged_categories = sorted(df["merged_category"].unique())
    category_to_id = {
        category: idx for idx, category in enumerate(merged_categories)
    }
    id_to_category = {
        idx: category for category, idx in category_to_id.items()
    }

    df["label"] = df["merged_category"].map(category_to_id).astype(int)
    df["label_name"] = df["merged_category"]

    return (
        df.reset_index(drop=True)[
            [
                "text",
                "label",
                "label_name",
                "category",
                "merged_category",
            ]
        ],
        id_to_category,
    )


def remove_conflicting_and_duplicate_texts(
    df: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, int, int]:
    duplicate_text_count = int(df.duplicated(subset=["text"]).sum())

    label_counts_per_text = df.groupby("text")[label_column].nunique()
    conflicting_texts = label_counts_per_text[
        label_counts_per_text > 1
    ].index

    conflicting_duplicate_count = int(len(conflicting_texts))

    df = df[~df["text"].isin(conflicting_texts)].copy()
    df = df.drop_duplicates(subset=["text"], keep="first")
    df = df.reset_index(drop=True)

    return df, duplicate_text_count, conflicting_duplicate_count


def remove_ag_train_test_overlap(
    ag_train_df: pd.DataFrame,
    ag_test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    test_texts = set(ag_test_df["text"])
    overlap_count = int(ag_train_df["text"].isin(test_texts).sum())

    ag_train_df = ag_train_df[
        ~ag_train_df["text"].isin(test_texts)
    ].copy()

    return ag_train_df.reset_index(drop=True), overlap_count


def create_splits(
    ag_train_clean: pd.DataFrame,
    ag_test_clean: pd.DataFrame,
    kaggle_clean: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    ag_train_df, ag_val_df = train_test_split(
        ag_train_clean,
        test_size=EXPERIMENT_CONFIG["ag_validation_size"],
        random_state=SEED,
        stratify=ag_train_clean["label"],
    )

    kaggle_train_val_df, kaggle_test_df = train_test_split(
        kaggle_clean,
        test_size=EXPERIMENT_CONFIG["kaggle_test_size"],
        random_state=SEED,
        stratify=kaggle_clean["label"],
    )

    kaggle_train_df, kaggle_val_df = train_test_split(
        kaggle_train_val_df,
        test_size=EXPERIMENT_CONFIG[
            "kaggle_validation_size_from_train_val"
        ],
        random_state=SEED,
        stratify=kaggle_train_val_df["label"],
    )

    return {
        "AG_News": {
            "train": ag_train_df.reset_index(drop=True),
            "validation": ag_val_df.reset_index(drop=True),
            "test": ag_test_clean.reset_index(drop=True),
        },
        "Kaggle_News": {
            "train": kaggle_train_df.reset_index(drop=True),
            "validation": kaggle_val_df.reset_index(drop=True),
            "test": kaggle_test_df.reset_index(drop=True),
        },
    }


def calculate_split_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[int, int, int]:
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    return (
        len(train_texts.intersection(val_texts)),
        len(train_texts.intersection(test_texts)),
        len(val_texts.intersection(test_texts)),
    )


def validate_no_split_overlap(
    datasets: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    leakage_rows = []

    for dataset_name, split_dict in datasets.items():
        train_val_overlap, train_test_overlap, val_test_overlap = (
            calculate_split_overlap(
                split_dict["train"],
                split_dict["validation"],
                split_dict["test"],
            )
        )

        leakage_rows.extend(
            [
                {
                    "dataset": dataset_name,
                    "overlap_type": "train_validation_text_overlap",
                    "count": train_val_overlap,
                },
                {
                    "dataset": dataset_name,
                    "overlap_type": "train_test_text_overlap",
                    "count": train_test_overlap,
                },
                {
                    "dataset": dataset_name,
                    "overlap_type": "validation_test_text_overlap",
                    "count": val_test_overlap,
                },
            ]
        )

        if train_val_overlap != 0:
            raise ValueError(
                f"{dataset_name} contains train-validation text overlap."
            )

        if train_test_overlap != 0:
            raise ValueError(
                f"{dataset_name} contains train-test text overlap."
            )

        if val_test_overlap != 0:
            raise ValueError(
                f"{dataset_name} contains validation-test text overlap."
            )

    return pd.DataFrame(leakage_rows)


def build_split_summary(
    datasets: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []

    for dataset_name, split_dict in datasets.items():
        for split_name, df in split_dict.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "rows": len(df),
                    "classes": df["label"].nunique(),
                }
            )

    return pd.DataFrame(rows)


def save_processed_datasets(
    datasets: dict[str, dict[str, pd.DataFrame]],
) -> None:
    file_names = {
        ("AG_News", "train"): "ag_train_processed.csv",
        ("AG_News", "validation"): "ag_validation_processed.csv",
        ("AG_News", "test"): "ag_test_processed.csv",
        ("Kaggle_News", "train"): "kaggle_train_processed.csv",
        ("Kaggle_News", "validation"): "kaggle_validation_processed.csv",
        ("Kaggle_News", "test"): "kaggle_test_processed.csv",
    }

    for (dataset_name, split_name), file_name in file_names.items():
        datasets[dataset_name][split_name][["text", "label"]].to_csv(
            PROCESSED_DATA_DIR / file_name,
            index=False,
        )


def save_label_mappings(kaggle_label_mapping: dict[int, str]) -> None:
    with open(
        PROCESSED_DATA_DIR / "ag_label_mapping.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(AG_LABEL_MAPPING, file, indent=4)

    with open(
        PROCESSED_DATA_DIR / "kaggle_label_mapping.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(kaggle_label_mapping, file, indent=4)


def save_preprocessing_summaries(
    split_summary: pd.DataFrame,
    leakage_summary: pd.DataFrame,
    duplicate_summary: pd.DataFrame,
    kaggle_merge_check: pd.DataFrame,
) -> None:
    split_summary.to_csv(
        RESULTS_DIR / "split_summary.csv",
        index=False,
    )

    leakage_summary.to_csv(
        RESULTS_DIR / "leakage_summary.csv",
        index=False,
    )

    duplicate_summary.to_csv(
        RESULTS_DIR / "duplicate_summary.csv",
        index=False,
    )

    kaggle_original_distribution = (
        kaggle_merge_check["category"]
        .value_counts()
        .rename_axis("original_category")
        .reset_index(name="count")
    )

    kaggle_merged_distribution = (
        kaggle_merge_check["merged_category"]
        .value_counts()
        .rename_axis("merged_category")
        .reset_index(name="count")
    )

    kaggle_original_distribution.to_csv(
        RESULTS_DIR / "kaggle_original_category_distribution.csv",
        index=False,
    )

    kaggle_merged_distribution.to_csv(
        RESULTS_DIR / "kaggle_merged_category_distribution.csv",
        index=False,
    )


def prepare_datasets() -> dict[str, dict[str, pd.DataFrame]]:
    ensure_directories_exist()
    ensure_raw_files_exist()

    ag_train_raw, ag_test_raw = load_ag_news()
    kaggle_raw = load_kaggle_news()

    validate_ag_news(
        ag_train_raw,
        ag_test_raw,
    )

    kaggle_merge_check = validate_kaggle_news(kaggle_raw)

    ag_train_clean = preprocess_ag_news(ag_train_raw)
    ag_test_clean = preprocess_ag_news(ag_test_raw)

    ag_train_clean, ag_duplicate_count, ag_conflicting_count = (
        remove_conflicting_and_duplicate_texts(
            ag_train_clean,
            label_column="label",
        )
    )

    ag_train_clean, ag_train_test_overlap_removed = (
        remove_ag_train_test_overlap(
            ag_train_clean,
            ag_test_clean,
        )
    )

    kaggle_clean, kaggle_label_mapping = preprocess_kaggle_news(kaggle_raw)

    kaggle_clean, kaggle_duplicate_count, kaggle_conflicting_count = (
        remove_conflicting_and_duplicate_texts(
            kaggle_clean,
            label_column="label",
        )
    )

    if ag_train_clean["label"].nunique() != EXPERIMENT_CONFIG[
        "ag_news_classes"
    ]:
        raise ValueError(
            "AG News training data does not contain the expected number of classes."
        )

    if ag_test_clean["label"].nunique() != EXPERIMENT_CONFIG[
        "ag_news_classes"
    ]:
        raise ValueError(
            "AG News test data does not contain the expected number of classes."
        )

    if kaggle_clean["label"].nunique() != EXPERIMENT_CONFIG[
        "kaggle_expected_merged_classes"
    ]:
        raise ValueError(
            "Kaggle News preprocessing produced an unexpected number of classes."
        )

    datasets = create_splits(
        ag_train_clean=ag_train_clean,
        ag_test_clean=ag_test_clean,
        kaggle_clean=kaggle_clean,
    )

    split_summary = build_split_summary(datasets)
    leakage_summary = validate_no_split_overlap(datasets)

    duplicate_summary = pd.DataFrame(
        [
            {
                "dataset": "AG_News",
                "metric": "duplicate_training_texts_detected_before_removal",
                "value": ag_duplicate_count,
            },
            {
                "dataset": "AG_News",
                "metric": "conflicting_training_texts_detected_before_removal",
                "value": ag_conflicting_count,
            },
            {
                "dataset": "AG_News",
                "metric": "train_test_overlaps_removed_from_training",
                "value": ag_train_test_overlap_removed,
            },
            {
                "dataset": "Kaggle_News",
                "metric": "duplicate_texts_detected_before_removal",
                "value": kaggle_duplicate_count,
            },
            {
                "dataset": "Kaggle_News",
                "metric": "conflicting_texts_detected_before_removal",
                "value": kaggle_conflicting_count,
            },
            {
                "dataset": "Kaggle_News",
                "metric": "rows_after_duplicate_removal",
                "value": len(kaggle_clean),
            },
        ]
    )

    save_processed_datasets(datasets)
    save_label_mappings(kaggle_label_mapping)
    save_preprocessing_summaries(
        split_summary=split_summary,
        leakage_summary=leakage_summary,
        duplicate_summary=duplicate_summary,
        kaggle_merge_check=kaggle_merge_check,
    )

    return datasets


if __name__ == "__main__":
    prepare_datasets()
