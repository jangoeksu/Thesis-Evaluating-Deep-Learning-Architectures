# Comparing CNN and RoBERTa for Multi-Class Classification of English News Articles

This repository accompanies the bachelor thesis *"Comparing CNN and RoBERTa for Multi-Class Classification of English News Articles: An Empirical Study on Accuracy, Efficiency, and Trade-Offs Using AG News and Kaggle Datasets"* submitted at the Vienna University of Economics and Business (WU Vienna).

The thesis presents an empirical comparison of convolutional neural networks (CNNs) and transformer-based models (RoBERTa) for news classification, with a focus on predictive performance, computational efficiency, and deployment-relevant trade-offs.

## Research Objectives

The study addresses the following research questions:

- How do CNN and RoBERTa differ in classification performance across datasets of varying complexity?
- How do the models compare in terms of training time, inference latency, and memory consumption?
- What trade-offs emerge between accuracy and efficiency, and how do these affect practical deployment decisions?


## Datasets

The experiments are conducted on two publicly available datasets:

- **AG News** (4 classes, balanced)
- **Kaggle News Category Dataset** (41 classes, fine-grained and imbalanced)

Due to licensing constraints, raw datasets are not included in this repository.  
Instructions for obtaining the data are provided in `data/README.md`.

## Repository Structure

