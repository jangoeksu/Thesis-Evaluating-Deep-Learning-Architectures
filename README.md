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

## Notes on Version Control

Raw datasets, processed datasets, trained model files, runtime logs, and generated run metadata are excluded from version control through `.gitignore`.

Selected final evaluation outputs may be committed to `results/` after the experiments have been successfully executed and reviewed.
