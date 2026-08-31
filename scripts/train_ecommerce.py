from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from shared_ml.ecommerce_transformers import ReviewTabularTransformer, ReviewTextTransformer

ROOT = Path(__file__).resolve().parents[1]
RAW_FIELDS = ["Summary", "Text", "HelpfulnessNumerator", "HelpfulnessDenominator"]
RANDOM_STATE = 42
SAMPLE_SIZE = 120_000
MAX_FEATURES = 12_000


def metric_row(y_true, prediction, score, seconds):
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "training_time_seconds": round(float(seconds), 2),
    }


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, list(model.classes_).index(1)]
    return model.decision_function(X)


def build_features():
    text = Pipeline([
        ("combine_text", ReviewTextTransformer()),
        ("tfidf", TfidfVectorizer(
            max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=3,
            max_df=0.98, sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
        )),
    ])
    tabular = Pipeline([
        ("engineer", ReviewTabularTransformer()),
        ("scale", MaxAbsScaler()),
    ])
    return FeatureUnion([("text", text), ("tabular", tabular)])


def clean_data(raw):
    counts = {"raw_rows": int(len(raw))}
    valid_score = raw["Score"].isin([1, 2, 3, 4, 5])
    counts["target_valid_rows"] = int(valid_score.sum())
    work = raw.loc[valid_score & raw["Score"].ne(3)].copy()
    counts["neutral_removed"] = int((raw["Score"] == 3).sum())
    work["target"] = work["Score"].isin([4, 5]).astype(int)
    work["Summary"] = work["Summary"].fillna("")
    work["Text"] = work["Text"].fillna("")
    nonblank = work["Text"].astype(str).str.strip().ne("")
    helpful = ((work["HelpfulnessNumerator"] >= 0) &
               (work["HelpfulnessDenominator"] >= 0) &
               (work["HelpfulnessNumerator"] <= work["HelpfulnessDenominator"]))
    counts["blank_text_removed"] = int((~nonblank).sum())
    counts["invalid_helpfulness_removed"] = int((~helpful).sum())
    work = work.loc[nonblank & helpful]
    duplicate = work.duplicated(subset=["ProductId", "UserId", "Score", "Summary", "Text"], keep="first")
    counts["duplicate_reviews_removed"] = int(duplicate.sum())
    work = work.loc[~duplicate].copy()
    counts["usable_rows"] = int(len(work))
    return work, counts


def save_eda(raw, usable, figdir):
    figdir.mkdir(parents=True, exist_ok=True)
    plots = [
        (raw["Score"].value_counts().sort_index(), "Original review scores", "Score", "Reviews", "score_distribution.png"),
        (usable["target"].map({0: "Negative", 1: "Positive"}).value_counts(), "Binary preference target", "Class", "Reviews", "target_distribution.png"),
    ]
    for series, title, xlabel, ylabel, filename in plots:
        ax = series.plot.bar(color=["#d56b45", "#177e78", "#e0a128", "#496f8a", "#7b5ea7"][:len(series)], figsize=(7, 4))
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.tight_layout(); plt.savefig(figdir / filename, dpi=160); plt.close()
    lengths = usable["Text"].str.split().str.len().clip(upper=500)
    ax = lengths.plot.hist(bins=50, color="#177e78", figsize=(7, 4))
    ax.set(title="Review length (clipped at 500 words for display)", xlabel="Words", ylabel="Reviews")
    plt.tight_layout(); plt.savefig(figdir / "review_length_distribution.png", dpi=160); plt.close()
    ratio = usable.HelpfulnessNumerator / usable.HelpfulnessDenominator.clip(lower=1)
    ax = ratio.plot.hist(bins=30, color="#e0a128", figsize=(7, 4))
    ax.set(title="Helpfulness ratio", xlabel="Helpful votes / total votes", ylabel="Reviews")
    plt.tight_layout(); plt.savefig(figdir / "helpfulness_distribution.png", dpi=160); plt.close()


def train():
    raw = pd.read_csv(ROOT / "data/ecommerce/Reviews.csv")
    usable, counts = clean_data(raw)
    save_eda(raw, usable, ROOT / "figures/ecommerce")
    modeling, _ = train_test_split(
        usable, train_size=min(SAMPLE_SIZE, len(usable)), stratify=usable["target"],
        random_state=RANDOM_STATE,
    )
    counts["modeling_rows"] = int(len(modeling))
    counts["modeling_distribution"] = {str(k): int(v) for k, v in modeling.target.value_counts().sort_index().items()}
    train, holdout = train_test_split(modeling, test_size=0.30, stratify=modeling.target, random_state=RANDOM_STATE)
    validation, test = train_test_split(holdout, test_size=0.50, stratify=holdout.target, random_state=RANDOM_STATE)
    X_train, y_train = train[RAW_FIELDS], train.target
    X_val, y_val = validation[RAW_FIELDS], validation.target
    X_test, y_test = test[RAW_FIELDS], test.target

    feature_builder = build_features()
    started = time.perf_counter(); Z_train = feature_builder.fit_transform(X_train, y_train); feature_time = time.perf_counter() - started
    Z_val = feature_builder.transform(X_val)
    text_dim = len(feature_builder.transformer_list[0][1].named_steps["tfidf"].vocabulary_)
    tab_dim = 5

    representation_results = {}
    representations = {
        "A_tabular_only": (Z_train[:, text_dim:], Z_val[:, text_dim:]),
        "B_text_only": (Z_train[:, :text_dim], Z_val[:, :text_dim]),
        "C_tabular_plus_text": (Z_train, Z_val),
    }
    for name, (a, b) in representations.items():
        model = LogisticRegression(max_iter=300, C=2.0, solver="liblinear", random_state=RANDOM_STATE)
        started = time.perf_counter(); model.fit(a, y_train); elapsed = time.perf_counter() - started
        pred = model.predict(b); representation_results[name] = metric_row(y_val, pred, scores(model, b), elapsed)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=300, C=2.0, solver="liblinear", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=24, min_samples_leaf=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=24, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE),
        "Linear SVC": LinearSVC(C=0.8, random_state=RANDOM_STATE),
        "SGD Classifier": SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1000, early_stopping=True, random_state=RANDOM_STATE),
        "Complement NB": ComplementNB(alpha=0.5),
    }
    comparison = {}; fitted = {}
    for name, model in candidates.items():
        started = time.perf_counter(); model.fit(Z_train, y_train); elapsed = time.perf_counter() - started
        pred = model.predict(Z_val)
        comparison[name] = metric_row(y_val, pred, scores(model, Z_val), elapsed)
        fitted[name] = model
        print(name, comparison[name], flush=True)

    best_f1 = max(row["f1"] for row in comparison.values())
    probability_candidates = [name for name, model in fitted.items() if hasattr(model, "predict_proba") and comparison[name]["f1"] >= best_f1 - 0.01]
    selected_name = max(probability_candidates, key=lambda name: (comparison[name]["f1"], comparison[name]["roc_auc"]))
    final_estimator = clone(candidates[selected_name])
    final_pipeline = Pipeline([("features", build_features()), ("model", final_estimator)])
    combined = pd.concat([train, validation], ignore_index=True)
    started = time.perf_counter(); final_pipeline.fit(combined[RAW_FIELDS], combined.target); final_fit_time = time.perf_counter() - started
    pred = final_pipeline.predict(X_test); score = scores(final_pipeline, X_test)
    test_metrics = metric_row(y_test, pred, score, final_fit_time)
    test_metrics["confusion_matrix"] = confusion_matrix(y_test, pred).tolist()
    test_metrics["classification_report"] = classification_report(y_test, pred, output_dict=True)

    modeldir = ROOT / "models/ecommerce"; modeldir.mkdir(parents=True, exist_ok=True)
    artifact = modeldir / "ecommerce_interest_model.joblib"; joblib.dump(final_pipeline, artifact, compress=3)
    loaded = joblib.load(artifact)
    demo = pd.DataFrame([
        {"Summary": "Excellent", "Text": "Fresh, tasty and exactly as described. I would buy it again.", "HelpfulnessNumerator": 1, "HelpfulnessDenominator": 1},
        {"Summary": "Disappointed", "Text": "Stale product, damaged package and a complete waste of money.", "HelpfulnessNumerator": 0, "HelpfulnessDenominator": 1},
        {"Summary": "Okay", "Text": "The taste is acceptable but the price is high for the size.", "HelpfulnessNumerator": 0, "HelpfulnessDenominator": 0},
    ])
    original_demo = final_pipeline.predict(demo); loaded_demo = loaded.predict(demo)
    reload_equal = bool(np.array_equal(original_demo, loaded_demo))

    fitted_features = final_pipeline.named_steps["features"]
    vectorizer = fitted_features.transformer_list[0][1].named_steps["tfidf"]
    final_vocabulary_size = len(vectorizer.vocabulary_)
    final_dimension = final_vocabulary_size + tab_dim
    actual_review = X_test.iloc[[0]]
    analyzer = vectorizer.build_analyzer(); tokens = analyzer(ReviewTextTransformer().transform(actual_review[["Summary", "Text"]])[0])[:20]
    token_ids = {token: int(vectorizer.vocabulary_[token]) for token in tokens if token in vectorizer.vocabulary_}
    vector = fitted_features.transformer_list[0][1].transform(actual_review[["Summary", "Text"]])
    nz = vector.nonzero()[1][:12]
    tfidf_values = [{"token": vectorizer.get_feature_names_out()[i], "index": int(i), "value": float(vector[0, i])} for i in nz]

    results = {
        "dataset": counts,
        "splits": {"train": len(train), "validation": len(validation), "test": len(test)},
        "raw_fields": RAW_FIELDS,
        "vocabulary_size": final_vocabulary_size,
        "tabular_dimension": tab_dim,
        "final_transformed_dimension": final_dimension,
        "feature_build_time_seconds": round(feature_time, 2),
        "representation_results": representation_results,
        "model_comparison": comparison,
        "selected_model": selected_name,
        "selection_rule": "Highest validation F1 among probability-capable models within 0.01 of the overall best F1; ROC-AUC breaks ties.",
        "test_metrics": test_metrics,
        "reload_predictions_equal": reload_equal,
        "demo_predictions": original_demo.tolist(),
        "representation_example": {"original": actual_review.iloc[0].to_dict(), "tokens": tokens, "token_ids": token_ids, "nonzero_tfidf": tfidf_values},
    }
    (modeldir / "training_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    figdir = ROOT / "figures/ecommerce"
    pd.DataFrame(representation_results).T[["f1", "roc_auc"]].plot.bar(figsize=(8, 4), rot=0); plt.tight_layout(); plt.savefig(figdir / "representation_comparison.png", dpi=160); plt.close()
    pd.DataFrame(comparison).T[["f1", "roc_auc"]].plot.bar(figsize=(9, 4), rot=25); plt.tight_layout(); plt.savefig(figdir / "model_comparison.png", dpi=160); plt.close()
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Negative", "Positive"], cmap="Blues"); plt.tight_layout(); plt.savefig(figdir / "confusion_matrix.png", dpi=160); plt.close()
    RocCurveDisplay.from_predictions(y_test, score); plt.tight_layout(); plt.savefig(figdir / "roc_curve.png", dpi=160); plt.close()
    coef = final_pipeline.named_steps["model"].coef_[0][:final_vocabulary_size]
    terms = vectorizer.get_feature_names_out(); top = np.r_[np.argsort(coef)[:10], np.argsort(coef)[-10:]]
    pd.Series(coef[top], index=terms[top]).sort_values().plot.barh(figsize=(8, 6), color=["#d56b45" if v < 0 else "#177e78" for v in coef[top]]); plt.tight_layout(); plt.savefig(figdir / "top_terms.png", dpi=160); plt.close()
    errors = test.loc[pred != y_test.to_numpy(), ["Summary", "Text", "Score", "target"]].copy().head(20)
    errors["prediction"] = pred[pred != y_test.to_numpy()][:20]
    errors.to_csv(modeldir / "error_examples.csv", index=False)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    train()
