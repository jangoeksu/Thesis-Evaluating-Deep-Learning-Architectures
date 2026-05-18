from __future__ import annotations

import gc
import json
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from transformers import set_seed

from configs.experiment_settings import (
    CONFIGS_DIR,
    DEVICE,
    LOGS_DIR,
    SEED,
)


def ensure_utility_directories_exist() -> None:
    """Create directories used for saved configurations and log files."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def reset_random_seeds(seed: int = SEED) -> None:
    """
    Reset random seeds and deterministic settings for reproducible experiments.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def is_mps_available() -> bool:
    """
    Return whether the installed PyTorch build can access Apple's MPS backend.
    """
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def cleanup_memory() -> None:
    """
    Trigger Python garbage collection and clear accelerator caches when possible.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if is_mps_available() and hasattr(torch, "mps"):
        empty_cache_function = getattr(torch.mps, "empty_cache", None)

        if callable(empty_cache_function):
            empty_cache_function()


def reset_peak_gpu_memory_stats() -> None:
    """
    Reset CUDA peak-memory counters when CUDA is available.

    Peak-memory tracking is currently collected only for CUDA devices.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def collect_peak_gpu_memory_metrics() -> dict[str, float | None]:
    """
    Return CUDA peak-memory metrics in megabytes, or `None` without CUDA.
    """
    if not torch.cuda.is_available():
        return {
            "peak_gpu_memory_allocated_mb": None,
            "peak_gpu_memory_reserved_mb": None,
        }

    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)

    return {
        "peak_gpu_memory_allocated_mb": float(peak_allocated_mb),
        "peak_gpu_memory_reserved_mb": float(peak_reserved_mb),
    }


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> dict[str, float]:
    """
    Compute core multi-class classification metrics used in the thesis.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def compute_roberta_metrics(eval_pred: Any) -> dict[str, float]:
    """
    Convert Hugging Face Trainer predictions into thesis evaluation metrics.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return compute_classification_metrics(
        y_true=labels,
        y_pred=predictions,
    )


def compute_inference_time_metrics(
    total_inference_time_seconds: float,
    number_of_samples: int,
) -> dict[str, float]:
    """
    Convert total inference time into total and per-sample timing metrics.
    """
    if number_of_samples <= 0:
        raise ValueError("number_of_samples must be greater than zero.")

    return {
        "total_inference_time_seconds": float(
            total_inference_time_seconds
        ),
        "inference_time_per_sample_ms": float(
            (total_inference_time_seconds / number_of_samples) * 1000
        ),
    }


def synchronize_cuda_if_available() -> None:
    """
    Synchronize active accelerator work before or after timing measurements.

    The original function name is retained for compatibility with existing
    experiment code. CUDA synchronization is used when available. If the run
    uses Apple's MPS backend and the installed PyTorch exposes synchronization,
    MPS synchronization is also attempted.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return

    if is_mps_available() and hasattr(torch, "mps"):
        synchronize_function = getattr(torch.mps, "synchronize", None)

        if callable(synchronize_function):
            synchronize_function()


def collect_system_info() -> dict[str, Any]:
    """
    Collect reproducibility-relevant software and hardware environment details.
    """
    cuda_available = torch.cuda.is_available()
    mps_available = is_mps_available()

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "active_device": str(DEVICE),
        "cuda_available": cuda_available,
        "cuda_device_count": (
            torch.cuda.device_count()
            if cuda_available
            else 0
        ),
        "cuda_gpu_name": (
            torch.cuda.get_device_name(0)
            if cuda_available
            else "No CUDA GPU available"
        ),
        "mps_available": mps_available,
        "mps_built": bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
        ),
    }


def save_json_file(
    path: str | Path,
    content: dict[str, Any],
) -> None:
    """
    Save a dictionary as an indented UTF-8 JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=4)


def append_jsonl(
    path: str | Path,
    row: dict[str, Any],
) -> None:
    """
    Append one JSON-serializable record to a JSON Lines file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")


def append_error_log(
    dataset_name: str,
    model_name: str,
    error: Exception,
) -> None:
    """
    Append a structured model-run error record to the experiment log.
    """
    append_jsonl(
        LOGS_DIR / "experiment_errors.jsonl",
        {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "model": model_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )


def append_run_config(
    dataset_name: str,
    model_name: str,
    config: dict[str, Any],
) -> None:
    """
    Append one structured model-run configuration to the experiment log.
    """
    append_jsonl(
        LOGS_DIR / "experiment_run_configs.jsonl",
        {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "model": model_name,
            "config": config,
        },
    )


def save_repository_run_metadata(
    run_id: str,
    run_config: dict[str, Any],
) -> None:
    """
    Save run configuration and environment metadata for reproducibility.
    """
    ensure_utility_directories_exist()

    system_info = collect_system_info()

    save_json_file(
        CONFIGS_DIR / f"{run_id}_experiment_config.json",
        run_config,
    )

    save_json_file(
        CONFIGS_DIR / f"{run_id}_system_info.json",
        system_info,
    )
