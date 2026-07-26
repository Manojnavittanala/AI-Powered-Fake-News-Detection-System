import json
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.preprocessing import clean_text, prepare_corpus

BASE_DIR = Path(__file__).resolve().parent


def _content_column(frame):
    title = frame['title'].fillna('').astype(str) if 'title' in frame else ''
    text = frame['text'].fillna('').astype(str) if 'text' in frame else ''
    if isinstance(title, str):
        return frame.iloc[:, 0].fillna('').astype(str)
    return (title + ' ' + text).str.strip()


def load_dataset(fake_path='dataset/Fake.csv', true_path='dataset/True.csv'):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df['label'] = 0
    true_df['label'] = 1
    fake_df['content'] = _content_column(fake_df)
    true_df['content'] = _content_column(true_df)
    data = pd.concat([fake_df[['content', 'label']], true_df[['content', 'label']]], ignore_index=True)
    data['content'] = prepare_corpus(data['content'])
    data = data[data['content'].str.len() > 0].drop_duplicates('content')
    return data.sample(frac=1, random_state=42).reset_index(drop=True)


def dataset_report(fake_path='dataset/Fake.csv', true_path='dataset/True.csv'):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    raw = pd.concat([fake_df.assign(label=0), true_df.assign(label=1)], ignore_index=True)
    content = _content_column(raw)
    cleaned = content.map(clean_text)
    tokens = [token for text in cleaned for token in text.split()]
    return {
        'dataset_size': int(len(raw)),
        'class_balance': {str(key): int(value) for key, value in raw['label'].value_counts().items()},
        'duplicate_articles': int(content.duplicated().sum()),
        'duplicate_clean_articles': int(cleaned.duplicated().sum()),
        'missing_values': {key: int(value) for key, value in raw.isna().sum().items()},
        'average_article_length': round(float(content.str.len().mean()), 2),
        'vocabulary_size': int(len(set(tokens))),
        'common_words': Counter(tokens).most_common(20),
        'label_conflicting_duplicates': int(
            raw.assign(content=content).groupby('content')['label'].nunique().gt(1).sum()
        ),
        'noise_rows': int((content.str.len() < 40).sum()),
    }


def _metrics(y_true, predictions, scores=None):
    result = {
        'accuracy': float(accuracy_score(y_true, predictions)),
        'precision': float(precision_score(y_true, predictions, zero_division=0)),
        'recall': float(recall_score(y_true, predictions, zero_division=0)),
        'f1_score': float(f1_score(y_true, predictions, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, predictions).tolist(),
        'classification_report': classification_report(y_true, predictions, zero_division=0),
    }
    result['roc_auc'] = float(roc_auc_score(y_true, scores)) if scores is not None and len(set(y_true)) == 2 else None
    return result


def _candidate_models():
    return {
        "logistic_regression": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=50000,
                    sublinear_tf=True
                )
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    C=1.0,
                    random_state=42
                )
            )
        ])
    }

def _decision_scores(model, texts):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(texts)[:, 1]
    if hasattr(model, 'decision_function'):
        return model.decision_function(texts)
    return None


def train():
    print("Dataset report:")

    report = dataset_report()

    print(json.dumps(report, indent=2))

    with open(BASE_DIR / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    data = load_dataset()

    X = data["content"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    candidates = _candidate_models()

    results = {}

    for name, candidate in candidates.items():

        print(f"\nTraining {name}...\n")

        candidate.fit(X_train, y_train)

        predictions = candidate.predict(X_test)

        scores = _decision_scores(candidate, X_test)

        results[name] = _metrics(
            y_test,
            predictions,
            scores
        )

    selected_name = "logistic_regression"

    selected_pipeline = candidates[selected_name]

    selected_pipeline.fit(X, y)

    selected_vectorizer = selected_pipeline.named_steps["tfidf"]

    selected_model = selected_pipeline.named_steps["model"]

    model_data = {
        "model": selected_model,
        "model_name": selected_name,
        "metrics": results[selected_name],
        "dataset_report": report,
        "preprocessing": "clean_text_v2_lemma"
    }

    with open(BASE_DIR / "model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    with open(BASE_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(selected_vectorizer, f)

    print("\n==============================")
    print("Training Complete")
    print("==============================")
    print(f"Selected Model : {selected_name}")
    print(f"Accuracy       : {results[selected_name]['accuracy']:.4f}")
    print(f"Precision      : {results[selected_name]['precision']:.4f}")
    print(f"Recall         : {results[selected_name]['recall']:.4f}")
    print(f"F1 Score       : {results[selected_name]['f1_score']:.4f}")
    print(f"ROC AUC        : {results[selected_name]['roc_auc']:.4f}")
    print(f"Vocabulary     : {len(selected_vectorizer.get_feature_names_out())}")

if __name__ == '__main__':
    train()
