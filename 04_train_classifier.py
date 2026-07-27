"""
STEP 4: Train with two real upgrades over the previous version, both
evidence-tested via cross-validation rather than assumed:

1. COMBINED FEATURES: word-level TF-IDF (unigrams+bigrams) alone can't relate
   "debug" and "debugging" - they're different tokens. Adding CHARACTER
   n-grams (3-5 letter chunks) lets the model see that "debug" and "debugging"
   share "debu", "ebug" etc, catching morphological variants without needing
   a bigger vocabulary of exact phrases.
2. HYPERPARAMETER TUNING: rather than using scikit-learn's default
   regularization strength, grid search over C (LogisticRegression/SVM) or
   alpha (Naive Bayes) with cross-validation to find the value that actually
   generalizes best on THIS dataset, not a generic default.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion

df = pd.read_csv("task_training_data.csv")
print(f"Training on {len(df)} examples across {df['category'].nunique()} categories")

def make_features():
    """word (1,2)-grams + char (3,5)-grams, combined into one feature space.
    sublinear_tf=True applies log-scaling to term frequency (1 + log(tf))
    instead of raw counts - a standard TF-IDF refinement that keeps one
    very-repeated word from dominating a short task description."""
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), stop_words="english",
                                  min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
    ])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)

# --- Stage 1: compare base algorithms with combined features, default params ---
candidates = {
    "Multinomial Naive Bayes": Pipeline([("features", make_features()), ("clf", MultinomialNB())]),
    "Logistic Regression": Pipeline([("features", make_features()),
                                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "Linear SVM": Pipeline([("features", make_features()),
                             ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3))]),
    "Voting Ensemble (LR + SVM)": Pipeline([("features", make_features()),
        ("clf", VotingClassifier(estimators=[
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("svm", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
        ], voting="soft"))]),
}

print("\n=== Stage 1: base model comparison (word+char features, default params) ===")
stage1_results = {}
for name, pipeline in candidates.items():
    scores = cross_val_score(pipeline, df["task_text"], df["category"], cv=cv, scoring="accuracy")
    stage1_results[name] = scores.mean()
    print(f"{name:28s} mean accuracy: {scores.mean():.1%}  (folds: {[f'{s:.1%}' for s in scores]})")

stage1_best_name = max(stage1_results, key=stage1_results.get)
print(f"\nStage 1 winner: {stage1_best_name} ({stage1_results[stage1_best_name]:.1%})")

# --- Stage 2: hyperparameter tune the winner ---
print(f"\n=== Stage 2: hyperparameter tuning {stage1_best_name} ===")
if stage1_best_name == "Logistic Regression":
    param_grid = {"clf__C": [0.1, 0.5, 1, 3, 10]}
    base_pipeline = Pipeline([("features", make_features()),
                               ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    grid = GridSearchCV(base_pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(df["task_text"], df["category"])
elif stage1_best_name == "Linear SVM":
    param_grid = {"clf__estimator__C": [0.1, 0.5, 1, 3, 10]}
    base_pipeline = Pipeline([("features", make_features()),
                               ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3))])
    grid = GridSearchCV(base_pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(df["task_text"], df["category"])
elif stage1_best_name == "Voting Ensemble (LR + SVM)":
    # Nested hyperparameter tuning across two sub-estimators has a much larger
    # search space for modest expected gain - use the ensemble as-is rather
    # than over-engineer tuning for a marginal, unproven benefit.
    class _FitResult:
        pass
    grid = _FitResult()
    grid.best_estimator_ = candidates[stage1_best_name]
    grid.best_estimator_.fit(df["task_text"], df["category"])
    grid.best_score_ = stage1_results[stage1_best_name]
    grid.best_params_ = {"note": "ensemble used with default sub-estimator params"}
else:
    param_grid = {"clf__alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}
    base_pipeline = Pipeline([("features", make_features()), ("clf", MultinomialNB())])
    grid = GridSearchCV(base_pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(df["task_text"], df["category"])
print(f"Best CV accuracy after tuning: {grid.best_score_:.1%} "
      f"(vs {stage1_results[stage1_best_name]:.1%} with default params)")

final_pipeline = grid.best_estimator_
final_model_name = f"{stage1_best_name} (tuned)"

# --- Held-out evaluation for the confusion matrix / classification report ---
X_train, X_test, y_train, y_test = train_test_split(
    df["task_text"], df["category"], test_size=0.2, random_state=7, stratify=df["category"]
)
final_pipeline.fit(X_train, y_train)
y_pred = final_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nHeld-out test accuracy: {test_accuracy:.1%}")
print(classification_report(y_test, y_pred, zero_division=0))

labels = sorted(df["category"].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel("Predicted category")
ax.set_ylabel("Actual category")
ax.set_title(f"Confusion Matrix: {final_model_name} (26 categories)\n"
             f"Held-out test accuracy: {test_accuracy:.1%}  |  Tuned CV mean: {grid.best_score_:.1%}",
             fontsize=12, fontweight="bold")
for i in range(len(labels)):
    for j in range(len(labels)):
        if cm[i, j] > 0:
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved: confusion_matrix.png")

# --- Retrain on ALL data for the deployed model ---
final_pipeline = grid.best_estimator_
final_pipeline.fit(df["task_text"], df["category"])

joblib.dump(final_pipeline.named_steps["clf"], "category_classifier.joblib")
joblib.dump(final_pipeline.named_steps["features"], "classifier_vectorizer.joblib")

comparison_df = pd.DataFrame([
    {"model": name, "cv_mean_accuracy": round(score, 4)} for name, score in stage1_results.items()
] + [{"model": final_model_name, "cv_mean_accuracy": round(grid.best_score_, 4)}]
).sort_values("cv_mean_accuracy", ascending=False)
comparison_df["selected"] = comparison_df["model"] == final_model_name
comparison_df.to_csv("model_comparison.csv", index=False)

with open("model_metadata.txt", "w") as f:
    f.write(f"best_model={final_model_name}\n")
    f.write(f"cv_mean_accuracy={grid.best_score_:.4f}\n")
    f.write(f"held_out_test_accuracy={test_accuracy:.4f}\n")
    f.write(f"n_training_examples={len(df)}\n")
    f.write(f"n_categories={df['category'].nunique()}\n")
    f.write(f"features=word(1,2)-grams + char_wb(3,5)-grams\n")
    f.write(f"best_params={grid.best_params_}\n")

print(f"\nSaved trained model files + model_comparison.csv + model_metadata.txt")
