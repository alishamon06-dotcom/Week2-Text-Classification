# Week 2 – Text Classification Strategies

## Project
SMS Spam Detection using TF-IDF with Multinomial Naive Bayes and Linear SVM.

This project demonstrates a reproducible binary text-classification workflow using the public UCI SMS Spam Collection. It compares two classical machine-learning baselines and reports measured results only after the script is run.

## Dataset
- Name: SMS Spam Collection
- Source: UCI Machine Learning Repository
- Page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- DOI: https://doi.org/10.24432/C5CC84
- Labels: `ham` (ordinary message) and `spam` (unwanted message)
- The dataset is downloaded by the script from the official UCI dataset page's public ZIP endpoint. If the endpoint changes, download the dataset manually from the UCI page and place the tab-separated file at `data/SMSSpamCollection`.

## Setup
Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python src/train_classifier.py
```

## What the script does
1. Downloads and loads the dataset.
2. Checks missing values and duplicate messages.
3. Uses a stratified 80/20 train-test split.
4. Builds word-level TF-IDF features (unigrams and bigrams).
5. Trains Multinomial Naive Bayes and Linear Support Vector Classification.
6. Evaluates both models with accuracy, precision, recall, F1-score, confusion matrix, and a classification report.
7. Saves the fitted pipeline and evaluation files under `reports/` and `models/`.

The TF-IDF vectorizer is fitted only on the training data through a scikit-learn Pipeline, preventing vocabulary leakage from the test set.

## Important interpretation
Spam detection is a binary classification task, not a sentiment-analysis task. Accuracy alone can hide errors on the smaller spam class, so precision, recall, and F1-score are also reported. The script prints measured scores from the local run; this README does not claim any unverified score.

## Project structure
- `src/train_classifier.py` – complete training and evaluation pipeline
- `notebooks/Week2_Text_Classification.ipynb` – guided notebook
- `requirements.txt` – Python dependencies
- `reports/` – generated metrics, confusion-matrix image and classification report
- `models/` – saved best pipeline
- `data/` – optional location for a manually downloaded dataset

## Suggested future improvements
Try character n-grams, class weighting, threshold tuning, cross-validation, calibrated probabilities, and a carefully evaluated transformer model. Keep a final untouched test set for the last comparison.

## Academic note
This is a learning project. Review the code, run it yourself, and add your own observations and measured outputs before submission. The proposed 32-hour schedule in the report is a planning estimate, not a claim that every hour has already been spent.
