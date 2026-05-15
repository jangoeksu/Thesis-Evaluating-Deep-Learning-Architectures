from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from configs.experiment_settings import (
    DEVICE,
    RESULTS_DIR,
)
from src.utils import (
    compute_classification_metrics,
    compute_inference_time_metrics,
    synchronize_cuda_if_available,
)

def evaluate_cnn(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_classification_metrics(
        y_true=all_labels,
        y_pred=all_predictions,
    )

    return metrics, all_labels, all_predictions


def test_cnn_with_timing(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    number_of_samples: int,
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()

    all_predictions = []
    all_labels = []

    synchronize_cuda_if_available()
    inference_start = time.time()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    synchronize_cuda_if_available()
    inference_time = time.time() - inference_start

    metrics = compute_classification_metrics(
        y_true=all_labels,
        y_pred=all_predictions,
    )

    metrics.update(
        compute_inference_time_metrics(
            total_inference_time_seconds=inference_time,
            number_of_samples=number_of_samples,
        )
    )

    return metrics, all_labels, all_predictions


def evaluate_roberta_validation(
    trainer: Any,
    validation_dataset: Any,
    validation_labels: list[int],
) -> tuple[dict[str, float], list[int], list[int]]:
    validation_output = trainer.predict(validation_dataset)
    validation_predictions = np.argmax(
        validation_output.predictions,
        axis=1,
    ).tolist()

    metrics = compute_classification_metrics(
        y_true=validation_labels,
        y_pred=validation_predictions,
    )

    return metrics, validation_labels, validation_predictions


def test_roberta_with_timing(
    trainer: Any,
    test_dataset: Any,
    test_labels: list[int],
    number_of_samples: int,
) -> tuple[dict[str, float], list[int], list[int]]:
    synchronize_cuda_if_available()
    inference_start = time.time()

    test_output = trainer.predict(test_dataset)

    synchronize_cuda_if_available()
    inference_time = time.time() - inference_start

    test_predictions = np.argmax(
        test_output.predictions,
        axis=1,
    ).tolist()

    metrics = compute_classification_metrics(
        y_true=test_labels,
        y_pred=test_predictions,
    )

    metrics.update(
        compute_inference_time_metrics(
            total_inference_time_seconds=inference_time,
            number_of_samples=number_of_samples,
        )
    )

    return metrics, test_labels, test_predictions


def create_classification_report_dataframe(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.reset_index()
    report_df = report_df.rename(columns={"index": "class_or_average"})

    return report_df


def create_confusion_matrix_dataframe(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
) -> pd.DataFrame:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in label_names],
        columns=[f"pred_{label}" for label in label_names],
    )

    return matrix_df


def create_prediction_dataframe(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
) -> pd.DataFrame:
    label_lookup = {
        label_id: label_name
        for label_id, label_name in enumerate(label_names)
    }

    predictions_df = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_label": y_pred,
        }
    )

    predictions_df["true_label_name"] = predictions_df["true_label"].map(
        label_lookup
    )
    predictions_df["predicted_label_name"] = predictions_df[
        "predicted_label"
    ].map(label_lookup)
    predictions_df["correct_prediction"] = (
        predictions_df["true_label"] == predictions_df["predicted_label"]
    )

    return predictions_df


def save_evaluation_outputs(
    dataset_name: str,
    model_name: str,
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
    run_id: str,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{run_id}_{dataset_name}_{model_name}".lower()

    classification_report_df = create_classification_report_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
    )

    confusion_matrix_df = create_confusion_matrix_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
    )

    predictions_df = create_prediction_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
    )

    classification_report_df.to_csv(
        RESULTS_DIR / f"{file_prefix}_classification_report.csv",
        index=False,
    )

    confusion_matrix_df.to_csv(
        RESULTS_DIR / f"{file_prefix}_confusion_matrix.csv",
        index=True,
    )

    predictions_df.to_csv(
        RESULTS_DIR / f"{file_prefix}_predictions.csv",
        index=False,
    )
