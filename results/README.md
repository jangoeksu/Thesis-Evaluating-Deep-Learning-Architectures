# Results

This folder stores generated preprocessing summaries and experiment outputs.

The files in this folder are created automatically when the preprocessing and training pipelines are executed. They are not required before running the project.

## Preprocessing summaries

Running:

```bash
python -m src.data
```

generates summary files such as:

```text
split_summary.csv
leakage_summary.csv
duplicate_summary.csv
kaggle_original_category_distribution.csv
kaggle_merged_category_distribution.csv
```

These files document the prepared datasets, including split sizes, class distributions, duplicate handling, and checks for text overlap between training, validation, and test splits.

## Experiment outputs

Running:

```bash
python -m src.train
```

generates evaluation files such as:

```text
latest_full_experiment_results.csv
<run_id>_results.csv
<run_id>_<dataset>_<model>_classification_report.csv
<run_id>_<dataset>_<model>_confusion_matrix.csv
<run_id>_<dataset>_<model>_predictions.csv
```

The consolidated result tables report classification performance, validation-based model selection metrics, training time, inference time, and peak GPU memory metrics where CUDA is available.

The classification reports provide class-level precision, recall, and F1-scores. The confusion matrices summarize classification errors across classes. The prediction files contain the true and predicted labels for the evaluated test instances.

## Version control

Selected final result files may be committed to the repository after the experiments have been successfully executed and reviewed. Runtime logs are excluded from version control through `.gitignore`.
