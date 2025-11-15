from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from scipy.stats import randint, uniform, loguniform

DATA_DIR = Path("data/raw")

DATA = {
    "car": {"file": "car.csv", "target": "class"},
    "aps": {"file": "aps.csv", "target": "class", "na_values": ["na"]},
    "covertype": {"file": "covertype.csv", "target": "Cover_Type"},
    "jannis": {"file": "jannis.csv", "target": "__target__"},
}

def _exists(name): 
    return (DATA_DIR / DATA[name]["file"]).exists()

class TargetLabelEncoder:
    def __init__(self):
        self._le = LabelEncoder()
        self.classes_ = None

    def fit(self, y):
        y = pd.Series(y) if not isinstance(y, (pd.Series, pd.Index)) else y
        self._le.fit(y.astype(str) if y.dtype == "object" else y)
        self.classes_ = list(self._le.classes_)
        return self

    def transform(self, y):
        y = pd.Series(y) if not isinstance(y, (pd.Series, pd.Index)) else y
        return self._le.transform(y.astype(str) if y.dtype == "object" else y)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y_encoded):
        return self._le.inverse_transform(np.asarray(y_encoded))

def encode_labels(y, encoder: TargetLabelEncoder | None = None):
    if encoder is None:
        enc = TargetLabelEncoder().fit(y)
        return enc.transform(y), enc
    else:
        return encoder.transform(y), encoder

def split_xy(df, target):
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()
        self.int_cols_ = X.select_dtypes(include=[np.integer]).columns.tolist()
        self.float_cols_ = X.select_dtypes(include=[np.floating]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        self.medians_ = X[self.int_cols_].median(numeric_only=True)
        self.means_ = X[self.float_cols_].mean(numeric_only=True)
        return self
    def transform(self, X):
        X = X.copy()
        if self.int_cols_:   
            X[self.int_cols_] = X[self.int_cols_].fillna(self.medians_)
        if self.float_cols_: 
            X[self.float_cols_] = X[self.float_cols_].fillna(self.means_)
        if self.cat_cols_:   
            X[self.cat_cols_] = X[self.cat_cols_].fillna("(NA)").astype("string")
        return X

def build_two_stage_preprocessor():
    stage1 = MissingValueHandler()
    enc_scale = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), make_column_selector(dtype_include=[np.number])),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    make_column_selector(dtype_include=["object","string","category"])),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline([("stage1_missing", stage1), ("stage2_encode_scale", enc_scale)])

def make_folds(y, n_splits=5, seed=42, out=None, name="dataset"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [{"train_idx": tr.tolist(), "valid_idx": va.tolist()} for tr,va in skf.split(np.zeros(len(y)), y)]
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        Path(out, f"{name}_skf{n_splits}.json").write_text(json.dumps(folds))
    return folds


def run_random_search(
    param_distributions,
    pipe,
    X, y,
    n_iter = 50,
    cv_splits = 10,
    random_state = 42,
    n_jobs = -1,
    scoring = "accuracy",
    verbose = True,
):

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scorer = make_scorer(accuracy_score) if scoring == "accuracy" else scoring

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        refit=True,
    )
    search.fit(X, y)

    results_df = (
        pd.DataFrame(search.cv_results_)
        .sort_values("mean_test_score", ascending=False)
        .reset_index(drop=True)
    )

    best_params_plain = {k.replace("xgb__", ""): v for k, v in search.best_params_.items()}
    best_score = float(search.best_score_)
    return best_params_plain, best_score, results_df, search

def run_bayes_search(
    search_spaces, 
    pipe,        
    X, y,
    n_iter=50,
    cv_splits=10,
    random_state=42,
    n_jobs=-1,
    scoring="accuracy",
    verbose=True,
    acq_func="EI",
):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scorer = make_scorer(accuracy_score) if scoring == "accuracy" else scoring

    search = BayesSearchCV(
        estimator = pipe,
        search_spaces = search_spaces,
        n_iter=n_iter,
        scoring=scorer,
        cv = cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        refit=True,
        optimizer_kwargs={"acq_func": acq_func},
    )
    search.fit(X, y)

    results_df = (
        pd.DataFrame(search.cv_results_)
        .sort_values("mean_test_score", ascending=False)
        .reset_index(drop=True)
    )
    best_params_plain = {k[5:]: v for k, v in search.best_params_.items()}
    best_score = float(search.best_score_)
    return best_params_plain, best_score, results_df, search
