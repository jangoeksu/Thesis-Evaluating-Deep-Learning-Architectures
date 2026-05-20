# Comparing CNN and RoBERTa for Multi-Class Classification of English News Articles

This repository contains the practical implementation, experiment pipeline, and selected final outputs for my bachelor thesis:

**Comparing CNN and RoBERTa for Multi-Class Classification of English News Articles: An Empirical Study on Accuracy, Efficiency, and Trade-Offs Using AG News and Kaggle Datasets**

The project investigates how a lightweight **Convolutional Neural Network (CNN)** and a pretrained **RoBERTa-base** transformer differ in English-language news classification. I designed and executed the experiment as a reproducible pipeline covering data preparation, model training, evaluation, statistical comparison, and post-hoc error analysis.

The repository includes both the code required to rerun the experiment and the selected outputs of the final completed run.

---

## Research Questions

The practical work is guided by the following research questions:

1. **How do CNN and RoBERTa differ in classification performance across news datasets of different complexity?**
2. **How do the models compare in terms of training time, inference time, and GPU memory consumption?**
3. **What trade-offs emerge between predictive performance and computational efficiency?**

The comparison is intentionally framed as a practical contrast between:

- a **CNN with embeddings trained from scratch** on the experiment datasets, and
- a **pretrained RoBERTa-base classifier** fine-tuned for the downstream classification task.

The study therefore evaluates a realistic lightweight-versus-pretrained-transformer trade-off rather than a perfectly symmetric architecture comparison.

---

## Datasets

### AG News

AG News is used as a comparatively simple four-class benchmark dataset with the classes:

- World
- Sports
- Business
- Sci/Tech

The raw train and test files are generated locally through the project download script from:

```text
fancyzhx/ag_news
```

### Kaggle News Category Dataset

The Kaggle News Category Dataset originally contains 42 topic categories. For this thesis, semantically related categories are merged into **22 broader classes**. The purpose of this step is to reduce label fragmentation while maintaining a substantially more complex multi-class classification task than AG News.

Examples of category merges include:

- `ARTS`, `ARTS & CULTURE`, `CULTURE & ARTS` → `ARTS_CULTURE`
- `BUSINESS`, `MONEY` → `BUSINESS_MONEY`
- `HEALTHY LIVING`, `WELLNESS` → `HEALTH_WELLNESS`
- `SCIENCE`, `TECH` → `SCIENCE_TECH`
- `STYLE`, `STYLE & BEAUTY` → `STYLE_BEAUTY`

The complete category merge mapping is defined in:

```text
configs/experiment_settings.py
```

The merged Kaggle dataset is especially relevant for the thesis because it introduces stronger class imbalance, greater semantic overlap between categories, and a noticeably more difficult classification setting.

---

## Final Completed Experiment Run

The final full experiment stored in this repository has the run ID:

```text
full_20260520_092957
```

The configuration and environment information for this run are preserved in:

```text
configs/full_20260520_092957_experiment_config.json
configs/full_20260520_092957_system_info.json
```

### Final Run Environment

The final run used:

- **Python:** 3.12.13
- **PyTorch:** 2.12.0+cu130
- **Transformers:** 5.8.1
- **Device:** CUDA
- **GPU:** NVIDIA A100-SXM4-80GB

The exact values are stored in the system-info JSON file.

---

## Final Results

The consolidated final result table is stored in:

```text
results/latest_full_experiment_results.csv
```

### Test-Set Performance

| Dataset | Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| AG News | CNN | 0.9096 | 0.9094 | 0.9094 |
| AG News | RoBERTa | 0.9501 | 0.9501 | 0.9501 |
| Kaggle News | CNN | 0.6318 | 0.5478 | 0.6226 |
| Kaggle News | RoBERTa | 0.7659 | 0.7165 | 0.7640 |

RoBERTa outperformed CNN on both datasets. The difference is moderate on AG News and substantially larger on the merged Kaggle News task.

### Efficiency Comparison

| Dataset | Model | Training Time | Inference Time per Sample |
|---|---:|---:|---:|
| AG News | CNN | 84.18 s | 0.075 ms |
| AG News | RoBERTa | 2054.51 s | 1.310 ms |
| Kaggle News | CNN | 119.82 s | 0.075 ms |
| Kaggle News | RoBERTa | 3022.62 s | 1.333 ms |

The central practical trade-off observed in the experiment is:

- **RoBERTa provides clearly stronger predictive performance.**
- **CNN is much faster and considerably lighter computationally.**

The measured runtime values refer to the documented final experiment environment and should not be interpreted as hardware-independent universal benchmarks.

---

## Statistical Significance Testing

To support the model comparison beyond point estimates, paired **McNemar tests** were calculated from the final test-set predictions.

The stored test outputs are:

```text
results/full_20260520_092957_ag_news_cnn_vs_roberta_mcnemar_test.csv
results/full_20260520_092957_kaggle_news_cnn_vs_roberta_mcnemar_test.csv
```

The tests indicate statistically significant differences in correctness patterns between CNN and RoBERTa on both datasets.

---

## Repository Structure

```text
Thesis-Evaluating-Deep-Learning-Architectures/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── __init__.py
│   ├── experiment_settings.py
│   ├── full_20260520_092957_experiment_config.json
│   └── full_20260520_092957_system_info.json
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── raw_data_metadata.json
│   └── processed/
│       └── .gitkeep
│
├── scripts/
│   ├── __init__.py
│   ├── download_data.py
│   └── run_experiment.py
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
│
├── results/
│   ├── README.md
│   ├── analysis/
│   ├── figures/
│   ├── duplicate_summary.csv
│   ├── leakage_summary.csv
│   ├── split_summary.csv
│   ├── kaggle_original_category_distribution.csv
│   ├── kaggle_merged_category_distribution.csv
│   ├── latest_full_experiment_results.csv
│   ├── full_20260520_092957_results.csv
│   ├── classification report CSV files
│   ├── confusion matrix CSV files
│   ├── prediction CSV files
│   ├── training-history CSV files
│   └── McNemar test CSV files
│
├── models/
│   └── .gitkeep
│
└── notebooks/
```

---

## Main Code Components

### `configs/experiment_settings.py`

This file defines the central experiment configuration, including:

- dataset paths,
- output paths,
- random seed,
- device selection,
- CNN hyperparameters,
- RoBERTa hyperparameters,
- dataset split sizes,
- Kaggle category merge map,
- fixed RoBERTa model revision.

### `scripts/download_data.py`

This script downloads the raw datasets and writes:

```text
data/raw/AG_train.csv
data/raw/AG_test.csv
data/raw/Kaggle_News.json
data/raw/raw_data_metadata.json
```

The metadata file also stores source information and SHA-256 hashes of the downloaded raw files.

### `src/data.py`

This module performs:

- dataset loading,
- schema validation,
- text cleaning,
- Kaggle category merging,
- duplicate handling,
- conflicting-label removal,
- AG News train-test overlap removal,
- stratified split generation,
- final leakage checks,
- preprocessing summary exports.

### `src/models.py`

This module defines:

- the CNN architecture,
- the initialization of the RoBERTa sequence-classification model.

### `src/train.py`

This module handles:

- CNN vocabulary construction,
- model-specific dataset wrappers,
- CNN training,
- RoBERTa training,
- validation-based model selection,
- saving model artifacts locally,
- experiment result aggregation,
- triggering McNemar tests once paired predictions are available.

### `src/evaluate.py`

This module handles:

- validation and test evaluation,
- direct inference-timing routines,
- classification reports,
- confusion matrices,
- row-level prediction exports,
- McNemar significance testing.

### `src/utils.py`

This module provides:

- seed resetting,
- reproducibility settings,
- GPU memory tracking,
- metric calculations,
- timing helpers,
- experiment metadata saving,
- structured logging utilities.

---

## Installation

The project uses pinned direct dependencies listed in:

```text
requirements.txt
```

Install them from the repository root with:

```bash
pip install -r requirements.txt
```

The final repository version was tested with Python 3.12.13 during the stored final run.

---

## Running the Full Experiment

From the repository root:

```bash
python -m scripts.run_experiment
```

This executes the complete workflow:

1. Download AG News from Hugging Face.
2. Download the Kaggle News Category Dataset through KaggleHub.
3. Store raw files locally in `data/raw/`.
4. Save raw-data metadata and file hashes.
5. Validate and preprocess both datasets.
6. Apply Kaggle category merging.
7. Remove duplicates, conflicting-label duplicates, and detected AG News train-test overlap.
8. Generate deterministic train, validation, and test splits.
9. Train CNN and RoBERTa on AG News.
10. Train CNN and RoBERTa on merged Kaggle News.
11. Evaluate all models.
12. Save reports, predictions, confusion matrices, training histories, significance tests, and final result tables.

---

## Running Only the Data Download

To download the raw datasets without starting training:

```bash
python -m scripts.download_data
```

This creates:

```text
data/raw/
├── AG_train.csv
├── AG_test.csv
├── Kaggle_News.json
└── raw_data_metadata.json
```

---

## Running Only the Preprocessing Pipeline

To run the data-cleaning and split-generation pipeline separately:

```bash
python -m src.data
```

This creates processed datasets locally and exports summary files to:

```text
data/processed/
results/
```

---

## Running the Training Pipeline Directly

To run preprocessing followed by the full model comparison:

```bash
python -m src.train
```

The training workflow trains and evaluates both models on both datasets and writes the final outputs automatically.

---

## Evaluation Outputs

### Main Result Tables

```text
results/latest_full_experiment_results.csv
results/full_20260520_092957_results.csv
```

### Classification Reports

The repository includes separate class-level reports for:

- AG News CNN,
- AG News RoBERTa,
- Kaggle News CNN,
- Kaggle News RoBERTa.

### Confusion Matrices

The repository includes matrix CSV files for all four model-dataset combinations.

### Prediction Files

The saved prediction files contain:

- true label,
- predicted label,
- true label name,
- predicted label name,
- correctness indicator.

These prediction files are also used for paired statistical comparison.

### Training Histories

Training-history CSV files store epoch-level information, including:

- CNN train loss and validation metrics,
- RoBERTa Trainer logs and evaluation metrics.

### McNemar Significance Tests

The repository contains significance-test summaries for:

- CNN vs. RoBERTa on AG News,
- CNN vs. RoBERTa on Kaggle News.

---

## Derived Analysis Outputs

### Figures

The directory:

```text
results/figures/
```

contains visualizations created from the final results, including:

- raw confusion matrix heatmaps,
- normalized confusion matrix heatmaps,
- macro-F1 comparison plot,
- training-time comparison plot,
- inference-time-per-sample comparison plot,
- Kaggle per-class F1 comparison plot,
- Kaggle per-class F1 gap plot.

### Aggregate Error Analysis Tables

The directory:

```text
results/analysis/
```

contains aggregate analysis tables, including:

- top confusion-pair CSVs for each dataset-model combination,
- Kaggle per-class F1 comparison table.

These files support the interpretation of:

- which classes are hardest to separate,
- which confusion patterns remain after fine-tuning RoBERTa,
- where RoBERTa improves most over CNN on the Kaggle task.

---

## Data Quality and Leakage Checks

The preprocessing pipeline explicitly records data-quality checks and split integrity.

The stored summaries include:

```text
results/duplicate_summary.csv
results/leakage_summary.csv
results/split_summary.csv
results/kaggle_original_category_distribution.csv
results/kaggle_merged_category_distribution.csv
```

The pipeline checks for:

- duplicate texts,
- texts assigned to conflicting labels,
- AG News train-test text overlap,
- train-validation overlap,
- train-test overlap,
- validation-test overlap.

These checks reduce the risk of data leakage and make the final comparison more reliable.

---

## Reproducibility

The repository includes several measures to improve reproducibility:

- fixed random seed,
- deterministic settings where supported,
- centralized experiment configuration,
- explicit train-validation-test splits,
- validation-based model selection,
- saved experiment configuration,
- saved system and hardware metadata,
- pinned direct dependencies,
- raw-data file hashes,
- final result tables and prediction outputs,
- significance-test outputs.

The final experiment is therefore reproducible in procedure and documented in detail. Exact bit-for-bit numerical identity across all hardware and software environments is not claimed.

---

## Scope of the Comparison

The empirical comparison should be interpreted carefully:

- RoBERTa benefits from large-scale pretraining.
- The CNN embeddings are trained from scratch.
- The experiment therefore compares two practically relevant modeling approaches rather than two equally pretrained architectures.
- The efficiency results are tied to the documented final A100 GPU environment.
- The Kaggle category merge map is explicit and reproducible, but some merged or retained categories remain semantically ambiguous.

These points are important for interpreting the performance differences and are discussed in the thesis.

---

## Version Control Scope

The repository intentionally includes:

- source code,
- experiment configuration,
- selected final run metadata,
- final result tables,
- reports,
- confusion matrices,
- predictions,
- training histories,
- significance-test outputs,
- preprocessing summaries,
- analysis figures,
- aggregate error-analysis tables.

The repository intentionally excludes:

- full raw dataset files,
- processed full datasets,
- trained model weights,
- local model checkpoints,
- runtime logs,
- local environments,
- temporary execution artifacts,
- text-based qualitative misclassification example files.

This keeps the repository compact while preserving the information needed to inspect the practical work and understand the reported thesis results.
