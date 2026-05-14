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

- **AG News**, used as a four-class benchmark dataset. The local train and test files correspond to the input data used by the Kaggle notebook `mohsinsial/ag-news-classifications`.
- **Kaggle News Category Dataset**, which contains 42 original categories and is merged into 22 broader target classes for this thesis.

Due to licensing and repository-size considerations, raw datasets are not included in this repository. Instructions for obtaining and placing the data are provided in `data/README.md`.

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
│   └── download_data.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
├── results/
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

## Data Setup

Raw dataset files are not included in this repository. They must be placed locally in:

```text
data/raw/
├── AG_train.csv
├── AG_test.csv
└── Kaggle_News.json
```

Further dataset setup details are provided in `data/README.md`.

## Running the Preprocessing Pipeline

To validate, clean, merge, split, and export the datasets, run:

```bash
python src/data.py
```

This creates processed datasets and preprocessing summaries in:

```text
data/processed/
results/
```

The preprocessing pipeline validates the expected dataset schemas, verifies class counts, applies the 22-class Kaggle category merge, removes problematic duplicate texts, checks for text overlap across dataset splits, and saves supporting summary files for documentation.

## Running the Full Experiment

To run the complete CNN–RoBERTa comparison on both datasets, execute:

```bash
python src/train.py
```

The full experiment pipeline trains and evaluates both models on AG News and the merged Kaggle News Category Dataset.

The resulting outputs include:

- consolidated result tables,
- classification reports,
- confusion matrices,
- row-level prediction exports,
- experiment metadata,
- timing metrics,
- GPU memory metrics when CUDA is available.

These outputs are written primarily to:

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

## Notes on Version Control

Raw datasets, processed datasets, trained model files, runtime logs, and generated run metadata are excluded from version control through `.gitignore`.

Selected final evaluation outputs may be committed to `results/` after the experiments have been successfully executed and reviewed.
