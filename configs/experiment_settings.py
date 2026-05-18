from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = RESULTS_DIR / "logs"


AG_TRAIN_PATH = RAW_DATA_DIR / "AG_train.csv"
AG_TEST_PATH = RAW_DATA_DIR / "AG_test.csv"
KAGGLE_NEWS_PATH = RAW_DATA_DIR / "Kaggle_News.json"


SEED = 42


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = select_device()


EXPERIMENT_CONFIG = {
    "project_name": "CNN_vs_RoBERTa_News_Classification",
    "seed": SEED,
    "device": str(DEVICE),
    "cnn_epochs": 5,
    "roberta_epochs": 5,
    "cnn_batch_size": 64,
    "roberta_batch_size": 16,
    "max_length": 256,
    "cnn_learning_rate": 1e-3,
    "roberta_learning_rate": 2e-5,
    "roberta_checkpoint": "roberta-base",
    "roberta_revision": None,
    "primary_metric": "macro_f1",
    "secondary_metric": "accuracy",
    "ag_news_classes": 4,
    "ag_validation_size": 0.10,
    "kaggle_original_classes": 42,
    "kaggle_category_merging": True,
    "kaggle_expected_merged_classes": 22,
    "kaggle_test_size": 0.15,
    "kaggle_validation_size_from_train_val": 0.1765,
}


BASELINE_CONFIG = {
    "model_name": "TFIDF_LogisticRegression",
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": None,
    "random_state": SEED,
}


CNN_CONFIG = {
    "embedding_dim": 128,
    "num_filters": 128,
    "kernel_sizes": (3, 4, 5),
    "dropout": 0.5,
    "max_vocab_size": 50000,
    "min_token_frequency": 2,
}


ROBERTA_CONFIG = {
    "checkpoint": "roberta-base",
    "revision": None,
    "weight_decay": 0.01,
    "logging_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "macro_f1",
    "greater_is_better": True,
    "save_total_limit": 1,
    "report_to": "none",
}


DATASET_VERSION_CONFIG = {
    "ag_news_source": "local AG_train.csv and AG_test.csv",
    "ag_news_revision": None,
    "kaggle_news_source": "local Kaggle_News.json",
    "kaggle_news_revision": None,
}


AG_LABEL_MAPPING = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


KAGGLE_CATEGORY_MERGE_MAP = {
    "ARTS": "ARTS_CULTURE",
    "ARTS & CULTURE": "ARTS_CULTURE",
    "CULTURE & ARTS": "ARTS_CULTURE",
    "BLACK VOICES": "IDENTITY_VOICES",
    "LATINO VOICES": "IDENTITY_VOICES",
    "QUEER VOICES": "IDENTITY_VOICES",
    "WOMEN": "WOMEN",
    "BUSINESS": "BUSINESS_MONEY",
    "MONEY": "BUSINESS_MONEY",
    "COLLEGE": "EDUCATION",
    "EDUCATION": "EDUCATION",
    "COMEDY": "COMEDY_WEIRD",
    "WEIRD NEWS": "COMEDY_WEIRD",
    "CRIME": "CRIME",
    "DIVORCE": "RELATIONSHIPS",
    "WEDDINGS": "RELATIONSHIPS",
    "ENTERTAINMENT": "ENTERTAINMENT",
    "MEDIA": "ENTERTAINMENT",
    "ENVIRONMENT": "ENVIRONMENT_GREEN",
    "GREEN": "ENVIRONMENT_GREEN",
    "FIFTY": "LIFESTYLE",
    "GOOD NEWS": "LIFESTYLE",
    "IMPACT": "LIFESTYLE",
    "FOOD & DRINK": "FOOD_DRINK",
    "TASTE": "FOOD_DRINK",
    "HEALTHY LIVING": "HEALTH_WELLNESS",
    "WELLNESS": "HEALTH_WELLNESS",
    "HOME & LIVING": "HOME_LIVING",
    "PARENTING": "PARENTING",
    "PARENTS": "PARENTING",
    "POLITICS": "POLITICS",
    "RELIGION": "RELIGION",
    "SCIENCE": "SCIENCE_TECH",
    "TECH": "SCIENCE_TECH",
    "SPORTS": "SPORTS",
    "STYLE": "STYLE_BEAUTY",
    "STYLE & BEAUTY": "STYLE_BEAUTY",
    "TRAVEL": "TRAVEL",
    "WORLD NEWS": "WORLD_NEWS",
    "U.S. NEWS": "WORLD_NEWS",
    "WORLDPOST": "WORLD_NEWS",
    "THE WORLDPOST": "WORLD_NEWS",
}
