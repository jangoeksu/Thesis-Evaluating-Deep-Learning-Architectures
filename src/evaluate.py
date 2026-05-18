from __future__ import annotations

import math
import time
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
    """Evaluate a CNN on a validation split without timing measurements."""
    model.eval()

    all_predictions: list[int] = []
    all_labels: list[int] = []

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
    """
    Evaluate the CNN test split and measure direct model-inference runtime.

    The timing covers the test-loop forward passes and associated tensor
    transfers already present in the experiment implementation.
    """
    model.eval()

    all_predictions: list[int] = []
    all_labels: list[int] = []

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
    """
    Evaluate RoBERTa on the validation set using the Hugging Face Trainer.

    This validation function is not used for runtime comparison, so the
    convenience of `trainer.predict` is retained here.
    """
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
    """
    Evaluate RoBERTa on the test split and measure direct forward-pass runtime.

    This avoids timing `trainer.predict`, which includes additional Trainer
    bookkeeping overhead and would be less comparable to the CNN timing loop.
    """
    model = trainer.model
    model.eval()

    test_dataloader = trainer.get_test_dataloader(test_dataset)
    all_predictions: list[int] = []

    synchronize_cuda_if_available()
    inference_start = time.time()

    with torch.no_grad():
        for batch in test_dataloader:
            model_inputs = {
                key: value.to(DEVICE)
                for key, value in batch.items()
                if key != "labels"
            }

            outputs = model(**model_inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy().tolist())

    synchronize_cuda_if_available()
    inference_time = time.time() - inference_start

    metrics = compute_classification_metrics(
        y_true=test_labels,
        y_pred=all_predictions,
    )

    metrics.update(
        compute_inference_time_metrics(
            total_inference_time_seconds=inference_time,
            number_of_samples=number_of_samples,
        )
    )

    return metrics, test_labels, all_predictions


def create_classification_report_dataframe(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
) -> pd.DataFrame:
    """Create a tabular classification report suitable for CSV export."""
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
    """Create a labeled confusion matrix DataFrame for thesis analysis."""
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
    """Create a row-level prediction table with label names and correctness."""
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
    """
    Save classification reports, confusion matrices, and row-level predictions.
    """
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


def validate_paired_prediction_inputs(
    y_true: list[int],
    model_a_predictions: list[int],
    model_b_predictions: list[int],
) -> None:
    """
    Validate that paired significance testing receives aligned predictions.
    """
    if len(y_true) != len(model_a_predictions):
        raise ValueError(
            "The number of true labels does not match model A predictions."
        )

    if len(y_true) != len(model_b_predictions):
        raise ValueError(
            "The number of true labels does not match model B predictions."
        )


def compute_mcnemar_test(
    y_true: list[int],
    model_a_predictions: list[int],
    model_b_predictions: list[int],
    model_a_name: str,
    model_b_name: str,
) -> dict[str, int | float | str]:
    """
    Compute a continuity-corrected McNemar test for two paired classifiers.

    The test evaluates whether the two classifiers differ significantly in
    their correctness patterns on the same test instances.
    """
    validate_paired_prediction_inputs(
        y_true=y_true,
        model_a_predictions=model_a_predictions,
        model_b_predictions=model_b_predictions,
    )

    model_a_only_correct = 0
    model_b_only_correct = 0

    for true_label, model_a_pred, model_b_pred in zip(
        y_true,
        model_a_predictions,
        model_b_predictions,
    ):
        model_a_correct = model_a_pred == true_label
        model_b_correct = model_b_pred == true_label

        if model_a_correct and not model_b_correct:
            model_a_only_correct += 1
        elif model_b_correct and not model_a_correct:
            model_b_only_correct += 1

    discordant_pairs = model_a_only_correct + model_b_only_correct

    if discordant_pairs == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        corrected_difference = max(
            abs(model_a_only_correct - model_b_only_correct) - 1,
            0,
        )
        statistic = (corrected_difference**2) / discordant_pairs
        p_value = math.erfc(math.sqrt(statistic / 2))

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "model_a_only_correct": model_a_only_correct,
        "model_b_only_correct": model_b_only_correct,
        "discordant_pairs": discordant_pairs,
        "mcnemar_statistic": float(statistic),
        "p_value": float(p_value),
        "test_variant": "continuity_corrected_asymptotic_mcnemar",
    }


def save_mcnemar_test_output(
    dataset_name: str,
    run_id: str,
    model_a_name: str,
    model_b_name: str,
    y_true: list[int],
    model_a_predictions: list[int],
    model_b_predictions: list[int],
) -> pd.DataFrame:
    """
    Compute and save a McNemar significance-test summary for two classifiers.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    significance_result = compute_mcnemar_test(
        y_true=y_true,
        model_a_predictions=model_a_predictions,
        model_b_predictions=model_b_predictions,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
    )

    significance_df = pd.DataFrame([significance_result])

    file_prefix = (
        f"{run_id}_{dataset_name}_{model_a_name}_vs_{model_b_name}"
    ).lower()

    significance_df.to_csv(
        RESULTS_DIR / f"{file_prefix}_mcnemar_test.csv",
        index=False,
    )

    return significance_df
