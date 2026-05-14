from __future__ import annotations

import inspect
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from configs.experiment_settings import (
    AG_LABEL_MAPPING,
    CNN_CONFIG,
    DEVICE,
    EXPERIMENT_CONFIG,
    MODELS_DIR,
    RESULTS_DIR,
    ROBERTA_CONFIG,
    SEED,
)
from src.data import prepare_datasets
from src.evaluate import (
    evaluate_cnn,
    evaluate_roberta_validation,
    save_evaluation_outputs,
    test_cnn_with_timing,
    test_roberta_with_timing,
)
from src.models import (
    TextCNN,
    initialize_roberta_model,
)
from src.utils import (
    append_error_log,
    append_run_config,
    cleanup_memory,
    collect_peak_gpu_memory_metrics,
    compute_roberta_metrics,
    reset_peak_gpu_memory_stats,
    reset_random_seeds,
    save_repository_run_metadata,
)


def simple_tokenize(text: str) -> list[str]:
    return str(text).lower().split()


def build_vocab(
    texts: list[str],
    max_vocab_size: int = CNN_CONFIG["max_vocab_size"],
    min_token_frequency: int = CNN_CONFIG["min_token_frequency"],
) -> dict[str, int]:
    token_frequencies: dict[str, int] = {}

    for text in texts:
        for token in simple_tokenize(text):
            token_frequencies[token] = token_frequencies.get(token, 0) + 1

    sorted_tokens = sorted(
        token_frequencies.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for token, count in sorted_tokens:
        if count >= min_token_frequency and len(vocab) < max_vocab_size:
            vocab[token] = len(vocab)

    return vocab


class CNNDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: dict[str, int],
        max_length: int,
    ) -> None:
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def encode_text(self, text: str) -> list[int]:
        tokens = simple_tokenize(text)[: self.max_length]

        token_ids = [
            self.vocab.get(token, self.vocab["<UNK>"])
            for token in tokens
        ]

        padding = [self.vocab["<PAD>"]] * (
            self.max_length - len(token_ids)
        )

        return token_ids + padding

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(
                self.encode_text(self.texts[index]),
                dtype=torch.long,
            ),
            "label": torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        }


class RobertaTextDataset(Dataset):
    def __init__(
        self,
        encodings: dict[str, list[list[int]]],
        labels: list[int],
    ) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return item


def train_cnn_one_epoch(
    model: TextCNN,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def save_cnn_artifacts(
    dataset_name: str,
    run_id: str,
    model: TextCNN,
    vocab: dict[str, int],
) -> None:
    dataset_model_dir = MODELS_DIR / f"{run_id}_{dataset_name}_CNN"
    dataset_model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        dataset_model_dir / "cnn_model_state.pt",
    )

    with open(
        dataset_model_dir / "cnn_vocab.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(vocab, file, indent=4)


def train_cnn(
    dataset_name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_classes: int,
    label_names: list[str],
    epochs: int,
    run_id: str,
) -> tuple[TextCNN, dict[str, float | None]]:
    reset_random_seeds(SEED)
    cleanup_memory()

    append_run_config(
        dataset_name=dataset_name,
        model_name="CNN",
        config={
            "run_id": run_id,
            "epochs": epochs,
            "batch_size": EXPERIMENT_CONFIG["cnn_batch_size"],
            "learning_rate": EXPERIMENT_CONFIG["cnn_learning_rate"],
            "max_length": EXPERIMENT_CONFIG["max_length"],
            "num_classes": num_classes,
            "embedding_dim": CNN_CONFIG["embedding_dim"],
            "num_filters": CNN_CONFIG["num_filters"],
            "kernel_sizes": CNN_CONFIG["kernel_sizes"],
            "dropout": CNN_CONFIG["dropout"],
            "max_vocab_size": CNN_CONFIG["max_vocab_size"],
            "min_token_frequency": CNN_CONFIG["min_token_frequency"],
        },
    )

    vocab = build_vocab(train_df["text"].tolist())

    train_dataset = CNNDataset(
        train_df,
        vocab,
        EXPERIMENT_CONFIG["max_length"],
    )

    validation_dataset = CNNDataset(
        validation_df,
        vocab,
        EXPERIMENT_CONFIG["max_length"],
    )

    test_dataset = CNNDataset(
        test_df,
        vocab,
        EXPERIMENT_CONFIG["max_length"],
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=EXPERIMENT_CONFIG["cnn_batch_size"],
        shuffle=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=EXPERIMENT_CONFIG["cnn_batch_size"],
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=EXPERIMENT_CONFIG["cnn_batch_size"],
        shuffle=False,
    )

    model = TextCNN(
        vocab_size=len(vocab),
        num_classes=num_classes,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=EXPERIMENT_CONFIG["cnn_learning_rate"],
    )

    criterion = nn.CrossEntropyLoss()

    best_validation_macro_f1 = -1.0
    best_model_state: dict[str, torch.Tensor] | None = None

    reset_peak_gpu_memory_stats()
    training_start = time.time()

    for _ in range(epochs):
        train_cnn_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
        )

        validation_metrics, _, _ = evaluate_cnn(
            model=model,
            dataloader=validation_loader,
        )

        if validation_metrics["macro_f1"] > best_validation_macro_f1:
            best_validation_macro_f1 = validation_metrics["macro_f1"]

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    training_time = time.time() - training_start
    memory_metrics = collect_peak_gpu_memory_metrics()

    if best_model_state is None:
        raise RuntimeError("CNN training did not produce a best model state.")

    model.load_state_dict(best_model_state)
    model.to(DEVICE)

    test_metrics, test_labels, test_predictions = test_cnn_with_timing(
        model=model,
        dataloader=test_loader,
        number_of_samples=len(test_df),
    )

    test_metrics["train_time_seconds"] = float(training_time)
    test_metrics["best_validation_macro_f1"] = float(
        best_validation_macro_f1
    )
    test_metrics.update(memory_metrics)

    save_evaluation_outputs(
        dataset_name=dataset_name,
        model_name="CNN",
        y_true=test_labels,
        y_pred=test_predictions,
        label_names=label_names,
        run_id=run_id,
    )

    save_cnn_artifacts(
        dataset_name=dataset_name,
        run_id=run_id,
        model=model,
        vocab=vocab,
    )

    return model, test_metrics


def tokenize_roberta_split(
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
) -> dict[str, list[list[int]]]:
    return tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=False,
        max_length=EXPERIMENT_CONFIG["max_length"],
    )


def initialize_roberta_trainer(
    model: Any,
    training_args: TrainingArguments,
    train_dataset: RobertaTextDataset,
    validation_dataset: RobertaTextDataset,
    tokenizer: AutoTokenizer,
    data_collator: DataCollatorWithPadding,
) -> Trainer:
    trainer_parameters = inspect.signature(Trainer.__init__).parameters

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_roberta_metrics,
    }

    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def build_roberta_training_arguments(
    output_dir: Path,
    epochs: int,
) -> TrainingArguments:
    training_args_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": EXPERIMENT_CONFIG[
            "roberta_batch_size"
        ],
        "per_device_eval_batch_size": EXPERIMENT_CONFIG[
            "roberta_batch_size"
        ],
        "learning_rate": EXPERIMENT_CONFIG[
            "roberta_learning_rate"
        ],
        "weight_decay": ROBERTA_CONFIG["weight_decay"],
        "logging_strategy": ROBERTA_CONFIG["logging_strategy"],
        "save_strategy": ROBERTA_CONFIG["save_strategy"],
        "load_best_model_at_end": ROBERTA_CONFIG[
            "load_best_model_at_end"
        ],
        "metric_for_best_model": ROBERTA_CONFIG[
            "metric_for_best_model"
        ],
        "greater_is_better": ROBERTA_CONFIG[
            "greater_is_better"
        ],
        "save_total_limit": ROBERTA_CONFIG[
            "save_total_limit"
        ],
        "seed": SEED,
        "data_seed": SEED,
        "fp16": bool(DEVICE.type == "cuda"),
        "report_to": ROBERTA_CONFIG["report_to"],
    }

    try:
        return TrainingArguments(
            **training_args_kwargs,
            eval_strategy="epoch",
        )
    except TypeError:
        return TrainingArguments(
            **training_args_kwargs,
            evaluation_strategy="epoch",
        )


def save_roberta_artifacts(
    dataset_name: str,
    run_id: str,
    trainer: Trainer,
    tokenizer: AutoTokenizer,
) -> None:
    dataset_model_dir = MODELS_DIR / f"{run_id}_{dataset_name}_RoBERTa"
    dataset_model_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(dataset_model_dir))
    tokenizer.save_pretrained(str(dataset_model_dir))


def train_roberta(
    dataset_name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_classes: int,
    label_names: list[str],
    epochs: int,
    run_id: str,
) -> tuple[Trainer, dict[str, float | None]]:
    reset_random_seeds(SEED)
    cleanup_memory()

    append_run_config(
        dataset_name=dataset_name,
        model_name="RoBERTa",
        config={
            "run_id": run_id,
            "epochs": epochs,
            "batch_size": EXPERIMENT_CONFIG["roberta_batch_size"],
            "learning_rate": EXPERIMENT_CONFIG[
                "roberta_learning_rate"
            ],
            "max_length": EXPERIMENT_CONFIG["max_length"],
            "checkpoint": EXPERIMENT_CONFIG["roberta_checkpoint"],
            "num_classes": num_classes,
            "weight_decay": ROBERTA_CONFIG["weight_decay"],
            "metric_for_best_model": ROBERTA_CONFIG[
                "metric_for_best_model"
            ],
            "load_best_model_at_end": ROBERTA_CONFIG[
                "load_best_model_at_end"
            ],
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        EXPERIMENT_CONFIG["roberta_checkpoint"]
    )

    train_encodings = tokenize_roberta_split(
        tokenizer=tokenizer,
        df=train_df,
    )

    validation_encodings = tokenize_roberta_split(
        tokenizer=tokenizer,
        df=validation_df,
    )

    test_encodings = tokenize_roberta_split(
        tokenizer=tokenizer,
        df=test_df,
    )

    train_dataset = RobertaTextDataset(
        train_encodings,
        train_df["label"].tolist(),
    )

    validation_dataset = RobertaTextDataset(
        validation_encodings,
        validation_df["label"].tolist(),
    )

    test_dataset = RobertaTextDataset(
        test_encodings,
        test_df["label"].tolist(),
    )

    model = initialize_roberta_model(
        num_classes=num_classes,
    ).to(DEVICE)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    output_dir = MODELS_DIR / f"{run_id}_{dataset_name}_RoBERTa_checkpoints"

    training_args = build_roberta_training_arguments(
        output_dir=output_dir,
        epochs=epochs,
    )

    trainer = initialize_roberta_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    reset_peak_gpu_memory_stats()
    training_start = time.time()

    trainer.train()

    training_time = time.time() - training_start
    memory_metrics = collect_peak_gpu_memory_metrics()

    validation_metrics, _, _ = evaluate_roberta_validation(
        trainer=trainer,
        validation_dataset=validation_dataset,
        validation_labels=validation_df["label"].tolist(),
    )

    test_metrics, test_labels, test_predictions = test_roberta_with_timing(
        trainer=trainer,
        test_dataset=test_dataset,
        test_labels=test_df["label"].tolist(),
        number_of_samples=len(test_df),
    )

    best_validation_macro_f1 = (
        float(trainer.state.best_metric)
        if trainer.state.best_metric is not None
        else float(validation_metrics["macro_f1"])
    )

    test_metrics["train_time_seconds"] = float(training_time)
    test_metrics["best_validation_macro_f1"] = best_validation_macro_f1
    test_metrics.update(memory_metrics)

    save_evaluation_outputs(
        dataset_name=dataset_name,
        model_name="RoBERTa",
        y_true=test_labels,
        y_pred=test_predictions,
        label_names=label_names,
        run_id=run_id,
    )

    save_roberta_artifacts(
        dataset_name=dataset_name,
        run_id=run_id,
        trainer=trainer,
        tokenizer=tokenizer,
    )

    return trainer, test_metrics


def build_dataset_registry(
    datasets: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, dict[str, Any]]:
    kaggle_label_names = (
        datasets["Kaggle_News"]["train"][["label", "label_name"]]
        .drop_duplicates()
        .sort_values("label")["label_name"]
        .tolist()
    )

    ag_label_names = [
        AG_LABEL_MAPPING[label_id]
        for label_id in sorted(AG_LABEL_MAPPING.keys())
    ]

    return {
        "AG_News": {
            "train": datasets["AG_News"]["train"],
            "validation": datasets["AG_News"]["validation"],
            "test": datasets["AG_News"]["test"],
            "num_classes": EXPERIMENT_CONFIG["ag_news_classes"],
            "label_names": ag_label_names,
        },
        "Kaggle_News": {
            "train": datasets["Kaggle_News"]["train"],
            "validation": datasets["Kaggle_News"]["validation"],
            "test": datasets["Kaggle_News"]["test"],
            "num_classes": EXPERIMENT_CONFIG[
                "kaggle_expected_merged_classes"
            ],
            "label_names": kaggle_label_names,
        },
    }


def create_full_experiment_config(
    run_id: str,
    dataset_registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_type": "full_experiment",
        "datasets": list(dataset_registry.keys()),
        "models": ["CNN", "RoBERTa"],
        "seed": SEED,
        "device": str(DEVICE),
        "cnn_epochs": EXPERIMENT_CONFIG["cnn_epochs"],
        "roberta_epochs": EXPERIMENT_CONFIG["roberta_epochs"],
        "cnn_batch_size": EXPERIMENT_CONFIG["cnn_batch_size"],
        "roberta_batch_size": EXPERIMENT_CONFIG["roberta_batch_size"],
        "cnn_learning_rate": EXPERIMENT_CONFIG["cnn_learning_rate"],
        "roberta_learning_rate": EXPERIMENT_CONFIG[
            "roberta_learning_rate"
        ],
        "roberta_checkpoint": EXPERIMENT_CONFIG["roberta_checkpoint"],
        "roberta_metric_for_best_model": ROBERTA_CONFIG[
            "metric_for_best_model"
        ],
        "max_length": EXPERIMENT_CONFIG["max_length"],
        "primary_metric": EXPERIMENT_CONFIG["primary_metric"],
        "secondary_metric": EXPERIMENT_CONFIG["secondary_metric"],
        "created_at": datetime.now().isoformat(),
    }


def create_result_row(
    run_id: str,
    dataset_name: str,
    model_name: str,
    status: str,
    dataset_info: dict[str, Any],
    metrics: dict[str, float | None] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}

    return {
        "run_id": run_id,
        "run_type": "full_experiment",
        "dataset": dataset_name,
        "model": model_name,
        "status": status,
        "seed": SEED,
        "device": str(DEVICE),
        "num_classes": dataset_info["num_classes"],
        "train_size": len(dataset_info["train"]),
        "validation_size": len(dataset_info["validation"]),
        "test_size": len(dataset_info["test"]),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "best_validation_macro_f1": metrics.get(
            "best_validation_macro_f1"
        ),
        "train_time_seconds": metrics.get("train_time_seconds"),
        "total_inference_time_seconds": metrics.get(
            "total_inference_time_seconds"
        ),
        "inference_time_per_sample_ms": metrics.get(
            "inference_time_per_sample_ms"
        ),
        "peak_gpu_memory_allocated_mb": metrics.get(
            "peak_gpu_memory_allocated_mb"
        ),
        "peak_gpu_memory_reserved_mb": metrics.get(
            "peak_gpu_memory_reserved_mb"
        ),
        "error_type": (
            type(error).__name__
            if error is not None
            else None
        ),
        "error_message": (
            str(error)
            if error is not None
            else None
        ),
        "completed_at": datetime.now().isoformat(),
    }


def save_full_results(
    full_results: list[dict[str, Any]],
    run_id: str,
) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(full_results)

    results_df.to_csv(
        RESULTS_DIR / f"{run_id}_results.csv",
        index=False,
    )

    results_df.to_csv(
        RESULTS_DIR / "latest_full_experiment_results.csv",
        index=False,
    )

    return results_df


def run_full_experiment() -> pd.DataFrame:
    reset_random_seeds(SEED)
    cleanup_memory()

    datasets = prepare_datasets()
    dataset_registry = build_dataset_registry(datasets)

    run_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    full_experiment_config = create_full_experiment_config(
        run_id=run_id,
        dataset_registry=dataset_registry,
    )

    save_repository_run_metadata(
        run_id=run_id,
        run_config=full_experiment_config,
    )

    full_results: list[dict[str, Any]] = []

    experiment_jobs = [
        (dataset_name, model_name)
        for dataset_name in dataset_registry.keys()
        for model_name in ["CNN", "RoBERTa"]
    ]

    for dataset_name, model_name in tqdm(
        experiment_jobs,
        desc="Full experiment progress",
        leave=True,
    ):
        dataset_info = dataset_registry[dataset_name]

        try:
            reset_random_seeds(SEED)
            cleanup_memory()

            if model_name == "CNN":
                trained_object, metrics = train_cnn(
                    dataset_name=dataset_name,
                    train_df=dataset_info["train"],
                    validation_df=dataset_info["validation"],
                    test_df=dataset_info["test"],
                    num_classes=dataset_info["num_classes"],
                    label_names=dataset_info["label_names"],
                    epochs=EXPERIMENT_CONFIG["cnn_epochs"],
                    run_id=run_id,
                )
            else:
                trained_object, metrics = train_roberta(
                    dataset_name=dataset_name,
                    train_df=dataset_info["train"],
                    validation_df=dataset_info["validation"],
                    test_df=dataset_info["test"],
                    num_classes=dataset_info["num_classes"],
                    label_names=dataset_info["label_names"],
                    epochs=EXPERIMENT_CONFIG["roberta_epochs"],
                    run_id=run_id,
                )

            result_row = create_result_row(
                run_id=run_id,
                dataset_name=dataset_name,
                model_name=model_name,
                status="success",
                dataset_info=dataset_info,
                metrics=metrics,
            )

            full_results.append(result_row)
            save_full_results(
                full_results=full_results,
                run_id=run_id,
            )

            del trained_object
            cleanup_memory()

        except Exception as error:
            append_error_log(
                dataset_name=dataset_name,
                model_name=model_name,
                error=error,
            )

            result_row = create_result_row(
                run_id=run_id,
                dataset_name=dataset_name,
                model_name=model_name,
                status="failed",
                dataset_info=dataset_info,
                error=error,
            )

            full_results.append(result_row)
            save_full_results(
                full_results=full_results,
                run_id=run_id,
            )

            cleanup_memory()

    return save_full_results(
        full_results=full_results,
        run_id=run_id,
    )


if __name__ == "__main__":
    run_full_experiment()
