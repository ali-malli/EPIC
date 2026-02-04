# EPIC
Enzyme-Product Interaction Classifier and Curve Predictor (EPIC)

EPIC is a lightweight ML framework for predicting enzyme–product interaction behaviors using β-glucosidase/glucose response as a case study.

This repository contains the **core training and inference code** for:
- **Classification**: glucose-response class prediction (ProstT5 + SVM + substrate concentration)
- **Regression**: activity–glucose response curve prediction (trigram frequencies + GradientBoostingRegressor)

> Note: This repo focuses on the **final models + feature generation**. 

---

## Repository structure

- `scripts/`
  - `embeddings.py` — generate ESM-2 and ProstT5 mean-pooled embeddings from FASTA
  - `ngrams.py` — generate normalized trigram frequency features (8000 trigrams)
  - `train_epic_classifier.py` — train final classifier on all labeled data and save a joblib bundle
  - `predict_epic_classifier.py` — run classifier inference on new enzymes (requires ProstT5 embeddings + substrate)
  - `train_epic_regressor.py` — train final regressor on all regression rows and save a joblib bundle
  - `predict_epic_regressor.py` — run regressor inference to predict activity vs glucose (requires trigrams + glucose + substrate)

- `data/` 
  - Place your datasets here, or provide paths when running scripts.

- `models/`
  - Output directory for saved `.joblib` model bundles.

---

## Installation

Create an environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate epic
