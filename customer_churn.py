# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

# set random state
RANDOM_STATE = 42
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"



#load data and clean
def load_telco(path: str) -> pd.DataFrame:
    """
    Load the Kaggle Telco churn CSV and standardize column names.

    normalize cols
    """
    df = pd.read_csv(path)

    # Normalize column names to a consistent, "safe" convention
    # strip whitespace
    # replace spaces with _
    # remove non-alphanumeric/underscore characters
    # tolower
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
        .str.lower()
    )
    return df


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean issues dataset.
    - trim whitespace on object columns
    - coerce totalcharges to numeric
    - seniorcitizen to numeric
    - drop duplicate customer IDs
    """
    df = df.copy()

    # normalize object/categorical columns: strip leading/trailing whitespace
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()


    #  blanks become NaN after coercion
    if "totalcharges" in df.columns:
        df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    
    # convert to yes/no to 1/0 for modeling
    if "churn" in df.columns:
        df["churn"] = df["churn"].map({"Yes": 1, "No": 0}).astype("Int64")

    # normalize to a nullable int
    if "seniorcitizen" in df.columns:
        df["seniorcitizen"] = df["seniorcitizen"].astype("Int64")

    # customerid is a unique identifier. Dedupe for row level identifier
    if "customerid" in df.columns:
        df = df.drop_duplicates(subset=["customerid"], keep="first")

    return df


def split_features_target(df: pd.DataFrame, target_col: str = "churn"):
    """
    split into X (features) and y (target).

    Critical:
    - drop customerid from features
    - encoding/imputation pipe
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    drop_cols = [target_col]
    if "customerid" in df.columns:
        drop_cols.append("customerid")

    X = df.drop(columns=drop_cols)
    y = df[target_col].astype(int)
    return X, y



#PREPROCESSING PIPELINE
def build_preprocessor_dense(X: pd.DataFrame) -> ColumnTransformer:
    """
    build a preprocessing ColumnTransformer - outputs dense matrix.

    What this does:
    - Numeric cols: median impute/standard scale
    - Cat cols: most-frequent impute/one-hot encode
    """
    #identify numeric columns by dtype
    num_cols = X.select_dtypes(include=[np.number, "Int64", "float64", "int64"]).columns.tolist()

    #everything else is treated as categorical
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(
        steps=[
            # fill missing numeric values with median
            ("imputer", SimpleImputer(strategy="median")),
            # Standardize numeric features (mean 0, std 1)
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            # missing categories with the most common category
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # one-hot encode categories into 0/1 columns
            # forces dense matrix output
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        # makes feature names cleaner
        verbose_feature_names_out=False,
    )
    return preprocessor



# MODEL EVAL (CV + HOLDOUT)
def evaluate_model_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> pd.DataFrame:
    """
    Cross-validate a pipeline with multiple metrics

    Metrics:
    - ROC-AUC: threshold-independent discrimination
    - avverage precision (PR-AUC): better for imbalanced classification than ROC-AUC sometimes
    - precision/recall/f1
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    res = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    #converts CV output dict to a readable format
    out = pd.DataFrame(res).drop(columns=["fit_time", "score_time"])
    summary = out.agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    return summary


def fit_and_report(pipe: Pipeline, X_train, X_test, y_train, y_test, threshold: float = 0.5) -> dict:
    """
    train on X_train and produce a holdout evaluation on X_test

    Why:
    - We use CV for robustness and a holdout set for a "final" sanity check

    threshold:
    - Controls churn classification.
    - 0.5 is default; in practice you tune threshold to match retention capacity/cost
    """
    pipe.fit(X_train, y_train)

    # Prefer predict_proba for probabilities; fallback to decision_function if needed
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
    else:
        scores = pipe.decision_function(X_test)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    preds = (proba >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "avg_precision": average_precision_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "classification_report": classification_report(y_test, preds, digits=4),
        "y_proba": proba,
        "y_pred": preds,
    }


def plot_roc_pr(y_true, y_proba, title_prefix: str):
    """
    plot ROC curve and Precision-Recall curve for a fitted model.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} - ROC Curve")
    plt.show()

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} - Precision-Recall Curve")
    plt.show()



#INTERPRETABILITY
def _get_preprocessor_and_model(pipe: Pipeline):
    """
    - preprocessor (ColumnTransformer)
    - final estimator (model)
    """
    pre = pipe.named_steps["pre"]
    mdl = pipe.named_steps["model"]
    return pre, mdl


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer):
    """
    Extract feature names after preprocessing (including one-hot encoded columns).
    """
    if hasattr(preprocessor, "get_feature_names_out"):
        names = preprocessor.get_feature_names_out()
        return pd.Index(names).astype(str).tolist()

    # fallback for old sklearn versions
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if transformer is None:
            continue

        if hasattr(transformer, "get_feature_names_out"):
            try:
                n = transformer.get_feature_names_out(cols)
                names.extend([str(x) for x in n])
            except Exception:
                n = transformer.get_feature_names_out()
                names.extend([str(x) for x in n])
        else:
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                names.extend([str(c) for c in cols])
            else:
                names.append(str(cols))

    return names


def logistic_coef_table(pipe: Pipeline, top_n: int = 25):
    """
    Build an interpretation table for LogisticRegression coefficients.
    """
    pre, mdl = _get_preprocessor_and_model(pipe)

    if mdl.__class__.__name__ != "LogisticRegression":
        raise ValueError("This function expects LogisticRegression as the final estimator.")

    feature_names = get_feature_names_from_preprocessor(pre)
    coefs = mdl.coef_.ravel()

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef_log_odds": coefs,
            "odds_ratio": np.exp(coefs),
            "abs_coef": np.abs(coefs),
        }
    ).sort_values("abs_coef", ascending=False)

    # positive coefficients increase churn prob
    top_pos = df.sort_values("coef_log_odds", ascending=False).head(top_n)

    # negative coefficients reduce churn prob
    top_neg = df.sort_values("coef_log_odds", ascending=True).head(top_n)

    return df, top_pos, top_neg


def plot_top_logreg_effects(top_pos: pd.DataFrame, top_neg: pd.DataFrame, title: str):
    """
    Plot the largest negative and positive coefficients for Logistic Regression.

    Why:
    - Useful in your report to communicate top churn drivers visually.
    """
    df_plot = pd.concat([top_neg, top_pos], axis=0).copy()

    plt.figure(figsize=(10, 10))
    plt.barh(df_plot["feature"].astype(str), df_plot["coef_log_odds"])
    plt.axvline(0, linestyle="--")
    plt.xlabel("Coefficient (log-odds)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def permutation_importance_table(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
    top_n: int = 30,
    scoring: str = "roc_auc",
):
    """
    Compute permutation importance on the holdout set.
    """
    pre, mdl = _get_preprocessor_and_model(pipe)
    feature_names = get_feature_names_from_preprocessor(pre)

    # transform  X_test into numeric matrix
    X_test_t = pre.transform(X_test)

    r = permutation_importance(
        mdl,
        X_test_t,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
            "abs_importance_mean": np.abs(r.importances_mean),
        }
    ).sort_values("abs_importance_mean", ascending=False)

    return out.head(top_n), out


def plot_top_importances(df_imp_top: pd.DataFrame, title: str):
    """
    Horizontal bar plot of top permutation importances.
    """
    df_plot = df_imp_top.iloc[::-1].copy()

    plt.figure(figsize=(10, 10))
    plt.barh(df_plot["feature"].astype(str), df_plot["importance_mean"])
    plt.axvline(0, linestyle="--")
    plt.xlabel("Mean importance (score decrease when permuted)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()



# END TO END PIPELINE
df_raw = load_telco(DATA_PATH)
df = clean_telco(df_raw)

print("Rows, cols:", df.shape)
print("Churn rate:", df["churn"].mean())

# data quality
missing_summary = df.isna().mean().sort_values(ascending=False).rename("missing_rate").to_frame()
print("\nTop missing rates:")
print(missing_summary.head(20))

# split into features/target (dropping customerid)
X, y = split_features_target(df, target_col="churn")

# stratified split -  churn rate is consistent
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# preprocessor - training feature set
pre = build_preprocessor_dense(X_train)

# models - logreg, randomforest, histgb
models = {
    "LogReg": LogisticRegression(max_iter=4000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=600,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    ),
    "HistGB": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

# models in the same preprocess/model pipeline
pipelines = {name: Pipeline(steps=[("pre", pre), ("model", mdl)]) for name, mdl in models.items()}

# cross-validate
print("\nCross-validated performance (train split):")
for name, pipe in pipelines.items():
    summary = evaluate_model_cv(pipe, X_train, y_train, cv_splits=5)
    print(f"\n{name}")
    print(summary)

# final holdout set
print("\nFinal holdout test performance:")
final_reports = {}
for name, pipe in pipelines.items():
    report = fit_and_report(pipe, X_train, X_test, y_train, y_test, threshold=0.5)
    final_reports[name] = report

    print(f"\n=== {name} (Test) ===")
    print(f"ROC-AUC: {report['roc_auc']:.4f}")
    print(f"Avg Precision (PR-AUC): {report['avg_precision']:.4f}")
    print("Confusion matrix:")
    print(report["confusion_matrix"])
    print("\nClassification report:")
    print(report["classification_report"])
    plot_roc_pr(y_test, report["y_proba"], title_prefix=name)

# logistic regression interpretation
logreg_pipe = pipelines["LogReg"]
logreg_pipe.fit(X_train, y_train)

coef_df, top_pos, top_neg = logistic_coef_table(logreg_pipe, top_n=20)

print("\nTop churn-increasing features (positive coefficient):")
print(top_pos[["feature", "coef_log_odds", "odds_ratio"]].to_string(index=False))

print("\nTop churn-decreasing features (negative coefficient):")
print(top_neg[["feature", "coef_log_odds", "odds_ratio"]].to_string(index=False))

plot_top_logreg_effects(top_pos, top_neg, title="Logistic Regression - Top Churn Drivers")


# key drivers
rf_pipe = pipelines["RandomForest"]
rf_pipe.fit(X_train, y_train)

rf_imp_top, rf_imp_all = permutation_importance_table(
    rf_pipe,
    X_test,
    y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
    top_n=25,
    scoring="roc_auc",
)
print("\nRandomForest - Top permutation importances (ROC-AUC impact):")
print(rf_imp_top[["feature", "importance_mean", "importance_std"]].to_string(index=False))
plot_top_importances(rf_imp_top, title="RandomForest - Permutation Importance (ROC-AUC)")

hgb_pipe = pipelines["HistGB"]
hgb_pipe.fit(X_train, y_train)

hgb_imp_top, hgb_imp_all = permutation_importance_table(
    hgb_pipe,
    X_test,
    y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
    top_n=25,
    scoring="roc_auc",
)
print("\nHistGB - Top permutation importances (ROC-AUC impact):")
print(hgb_imp_top[["feature", "importance_mean", "importance_std"]].to_string(index=False))
plot_top_importances(hgb_imp_top, title="HistGB - Permutation Importance (ROC-AUC)")
