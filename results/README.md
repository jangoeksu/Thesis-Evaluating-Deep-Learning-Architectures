# Comparing CNN and RoBERTa for Multi-Class Classification of English News Articles

This repository accompanies the bachelor thesis *"Comparing CNN and RoBERTa for Multi-Class Classification of English News Articles: An Empirical Study on Accuracy, Efficiency, and Trade-Offs Using AG News and Kaggle Datasets"* submitted at the Vienna University of Economics and Business (WU Vienna).

The thesis presents an empirical comparison of convolutional neural networks (CNNs) and transformer-based models (RoBERTa) for English-language news article classification. The comparison focuses on predictive performance, computational efficiency, and deployment-relevant trade-offs.

## Research Objectives

The study addresses the following research questions:

- How do CNN and RoBERTa differ in classification performance across datasets of varying complexity?
- How do the models compare in terms of training time, inference latency, and GPU memory consumption?
- What trade-offs emerge between accuracy and efficiency, and how do these affect practical deployment decisions?

## Datasets

The experiments are conducted on two publicly available datasets:

- **AG News**, used as a four-class benchmark dataset. The raw train and test files are generated locally through `scripts/download_data.py` from the Hugging Face dataset `fancyzhx/ag_news`.
- **Kaggle News Category Dataset**, which contains 42 original categories and is merged into 22 broader target classes for this thesis. The raw JSON file is downloaded automatically through `scripts/download_data.py`.

Due to licensing and repository-size considerations, raw datasets are not included in this repository. They are generated or downloaded locally through the project setup script.

## Repository Structure

```text
repo/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── __init__.py
│   ├── download_data.py
│   └── run_experiment.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
├── results/
│   └── README.md
├── configs/
│   ├── __init__.py
│   └── experiment_settings.py
└── models/
```

## Environment

The project is intended to be run with Python 3.10 or newer.

Install the required Python packages from the repository root:

```bash
pip install -r requirements.txt
```

The Kaggle News Category Dataset is downloaded through KaggleHub. Depending on the local environment, Kaggle authentication may need to be configured once before the first download.

## One-Command Full Experiment

After installing the dependencies and configuring Kaggle authentication if required, the complete experiment can be executed from the repository root with:

```bash
python -m scripts.run_experiment
```

This command performs the full workflow:

```text
1. Download AG News from Hugging Face.
2. Download the Kaggle News Category Dataset through KaggleHub.
3. Store all raw files in data/raw/.
4. Run preprocessing, validation, category merging, duplicate handling, and split generation.
5. Train CNN and RoBERTa on both datasets.
6. Save evaluation outputs, model artifacts, and experiment summaries.
```

## Data Setup Only

To download and prepare the raw datasets without starting the full experiment, run:

```bash
python -m scripts.download_data
```

This creates:

```text
data/raw/
├── AG_train.csv
├── AG_test.csv
└── Kaggle_News.json
```

Further information about the datasets and generated files is provided in `data/README.md`.

## Running the Preprocessing Pipeline Separately

To validate, clean, merge, split, and export the datasets without starting model training, run:

```bash
python -m src.data
```

This creates processed datasets and preprocessing summaries in:

```text
data/processed/
results/
```

The preprocessing pipeline validates the expected dataset schemas, verifies class counts, applies the 22-class Kaggle category merge, removes problematic duplicate texts, checks for text overlap across dataset splits, and saves supporting summary files for documentation.

## Running the Training Pipeline Separately

To run the complete CNN–RoBERTa comparison after the raw data files are available, execute:

```bash
python -m src.train
```

The training pipeline automatically calls the preprocessing step before model training. It then trains and evaluates both models on AG News and the merged Kaggle News Category Dataset.

## Outputs

The resulting outputs include:

- consolidated result tables,
- classification reports,
- confusion matrices,
- row-level prediction exports,
- preprocessing summaries,
- experiment metadata,
- timing metrics,
- peak GPU memory metrics when CUDA is available.

Evaluation outputs are written primarily to:

```text
results/
```

Model artifacts and checkpoints are written locally to:

```text
models/
```

Generated model files are excluded from version control through `.gitignore`.

## Reproducibility

The repository uses a fixed random seed and stores experiment settings centrally in `configs/experiment_settings.py`.

The CNN training procedure selects the best model based on validation macro-F1. RoBERTa is configured to load the best checkpoint based on validation macro-F1 as well, ensuring a more consistent model-selection strategy across architectures.

The final comparison records accuracy, macro-F1, weighted-F1, training time, inference time, and peak GPU memory metrics where CUDA is available.

## Timing Metrics

Training time includes the complete model-fitting phase and, where applicable, validation-based model selection during training.

Inference time is measured over the full evaluation pass on the test set and is reported both as:

- total inference time,
- average inference time per sample.

When GPU execution is available, CUDA synchronization is applied before and after timed operations to improve timing accuracy.

## Interpretation of the CNN–RoBERTa Comparison

The CNN and RoBERTa models are compared as alternative practical approaches for multi-class news classification under a shared preprocessing, evaluation, and reporting pipeline.

However, the comparison is not a strictly equal-resource architecture comparison. RoBERTa benefits from large-scale prior language-model pretraining, whereas the CNN baseline is trained from randomly initialized embeddings within this experiment.

The results are therefore interpreted as a comparison of practical model choices and their trade-offs in classification quality, computational effort, and inference efficiency, rather than as an isolated comparison of neural architecture alone.

## CNN Tokenization Choice

The CNN baseline uses a lightweight lowercasing and regex-based word tokenization strategy that preserves apostrophe-containing tokens while remaining computationally simple and reproducible.

The tokenizer is intentionally simpler than the subword tokenizer used by RoBERTa. This difference reflects the broader practical trade-off examined in the thesis: a lightweight conventional neural architecture versus a stronger pretrained transformer-based pipeline.

## Kaggle Category Consolidation

The Kaggle News Category Dataset contains 42 original labels. For this thesis, these labels are consolidated into 22 broader target classes, as defined in `configs/experiment_settings.py`.

This reduction serves two purposes:

1. It reduces category sparsity and improves the feasibility of multi-class comparison.
2. It creates broader topical classes that are more suitable for a compact bachelor thesis experiment.

Some merge decisions are pragmatic modeling choices rather than universally fixed taxonomies. For example, the mapping of `U.S. NEWS` into `WORLD_NEWS` is an explicit grouping decision made for this project and documented in the configuration file.

## Exact Environment Reproduction

The main `requirements.txt` file provides the required package set for installing and running the project.

For exact reproduction of the final thesis experiment environment, the resolved package versions used during the final execution should be exported after setup via:

```bash
pip freeze > requirements-lock.txt
```

This file records the precise package versions used for the final experimental run.

## Notes on Version Control

Raw datasets, processed datasets, trained model files, runtime logs, and generated run metadata are excluded from version control through `.gitignore`.

Selected final evaluation outputs may be committed to `results/` after the experiments have been successfully executed and reviewed.
