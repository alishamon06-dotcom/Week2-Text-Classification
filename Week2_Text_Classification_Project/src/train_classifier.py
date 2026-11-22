"""
Week 2: SMS spam text classification.
Downloads the UCI SMS Spam Collection, trains NB and Linear SVM pipelines,
and saves measured evaluation outputs. No performance scores are hard-coded.
"""
from pathlib import Path
import io
import zipfile
import requests
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
LOCAL_FILE = DATA_DIR / "SMSSpamCollection"

def load_dataset():
    if LOCAL_FILE.exists():
        raw = pd.read_csv(LOCAL_FILE, sep="\\t", header=None, names=["label", "text"], encoding="utf-8")
    else:
        print("Downloading the public UCI SMS Spam Collection...")
        response = requests.get(UCI_ZIP_URL, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            member = next((n for n in archive.namelist() if n.endswith("SMSSpamCollection")), None)
            if member is None:
                raise FileNotFoundError("SMSSpamCollection was not found in the UCI archive.")
            with archive.open(member) as stream:
                raw = pd.read_csv(stream, sep="\\t", header=None, names=["label", "text"], encoding="utf-8")
    return raw

def main():
    df = load_dataset()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["label"].isin(["ham", "spam"])].copy()

    missing_before = int(df[["label", "text"]].isna().sum().sum())
    duplicate_count = int(df.duplicated(subset=["label", "text"]).sum())
    df = df.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    print(f"Rows after duplicate removal: {len(df)}")
    print(f"Missing cells before text fill: {missing_before}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print("\\nClass distribution:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=0.20, random_state=42, stratify=df["target"]
    )

    models = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=2,
                                      sublinear_tf=True)),
            ("classifier", MultinomialNB(alpha=0.5))
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=2,
                                      sublinear_tf=True)),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42))
        ])
    }

    results = []
    best_name, best_f1, best_model = None, -1, None
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision_spam": precision_score(y_test, predictions, zero_division=0),
            "recall_spam": recall_score(y_test, predictions, zero_division=0),
            "f1_spam": f1_score(y_test, predictions, zero_division=0)
        }
        results.append(metrics)
        print(f"\\n{name}")
        print(pd.Series(metrics).to_string())
        print("\\nClassification report:")
        print(classification_report(y_test, predictions, target_names=["ham", "spam"], zero_division=0))

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        display = ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"])
        display.plot(values_format="d")
        plt.title(f"Confusion Matrix – {name}")
        safe_name = "naive_bayes" if "Naive" in name else "linear_svm"
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"confusion_matrix_{safe_name}.png", dpi=160)
        plt.close()

        (REPORT_DIR / f"classification_report_{safe_name}.txt").write_text(
            classification_report(y_test, predictions, target_names=["ham", "spam"], zero_division=0),
            encoding="utf-8"
        )
        if metrics["f1_spam"] > best_f1:
            best_name, best_f1, best_model = name, metrics["f1_spam"], model

    pd.DataFrame(results).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    joblib.dump(best_model, MODEL_DIR / "best_text_classifier.joblib")
    summary = (
        f"Dataset rows used: {len(df)}\\n"
        f"Train rows: {len(X_train)}\\nTest rows: {len(X_test)}\\n"
        f"Selected by spam F1: {best_name}\\n"
        f"Measured spam F1: {best_f1:.4f}\\n"
        "Selection is based on this fixed hold-out split; use cross-validation for a more robust comparison.\\n"
    )
    (REPORT_DIR / "run_summary.txt").write_text(summary, encoding="utf-8")
    print("\\n" + summary)
    print(f"Files saved in: {REPORT_DIR}")

if __name__ == "__main__":
    main()
