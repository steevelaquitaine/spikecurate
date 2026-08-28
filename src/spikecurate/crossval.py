"""Cross-validated fitting/evaluation of a GLM (fractional logistic) classifier."""
import copy
import logging
import random

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

LABEL_COL = "quality_label"


def _fit_glm(formula: str, data: pd.DataFrame):
    """Fit an unregularized binomial GLM."""
    return sm.GLM.from_formula(formula, family=sm.families.Binomial(), data=data).fit()


def _split_train_test(dataset: pd.DataFrame, split_ratio: float, seed: int):
    """Split `dataset` into (train, test) by row position, seeded for reproducibility."""
    random.seed(seed)
    n_train = int(np.round(split_ratio * dataset.shape[0]))
    indices = list(range(dataset.shape[0]))
    train_idx = random.sample(indices, n_train)
    test_idx = sorted(set(indices) - set(train_idx))
    return dataset.iloc[train_idx, :], dataset.iloc[test_idx, :]


def _scale(train: pd.DataFrame, test: pd.DataFrame, predictors: list):
    """Z-score `predictors` columns, fit on `train` only, applied to both."""
    scaler = StandardScaler()
    train = train.copy()
    test = test.copy()
    train[predictors] = scaler.fit_transform(train[predictors])
    test[predictors] = scaler.transform(test[predictors])
    return train, test


def single_fold_metrics(
    model_formula: str,
    dataset: pd.DataFrame,
    split_ratio: float = 0.75,
    seed: int = 0,
    thresh: float = 0.8,
    scale_data: bool = False,
) -> dict:
    """Precision/recall of one random train/test split."""
    assert LABEL_COL in dataset.columns, f"dataset must contain a '{LABEL_COL}' column"

    train_dataset, test_dataset = _split_train_test(dataset, split_ratio, seed)

    if scale_data:
        predictors = [c for c in dataset.columns if c != LABEL_COL]
        train_dataset, test_dataset = _scale(train_dataset, test_dataset, predictors)

    result = _fit_glm(model_formula, train_dataset)

    test_label = test_dataset[LABEL_COL]
    test_dataset = test_dataset.drop(columns=[LABEL_COL])

    features = result.params.index[1:]
    test_features = test_dataset.loc[:, features]
    test_features.insert(0, "intercept", 1)

    predictions = (result.predict(test_features) >= thresh).astype(int)
    return {
        "precision": metrics.precision_score(test_label, predictions, zero_division=0),
        "recall": metrics.recall_score(test_label, predictions, zero_division=0),
    }


def crossval_metrics(
    dataset: pd.DataFrame,
    model_formula: str,
    split_ratio: float = 0.75,
    seeds: np.ndarray = None,
    thresh: float = 0.8,
    scale_data: bool = False,
) -> np.ndarray:
    """Precision/recall across many random train/test splits (one per seed)."""
    seeds = np.arange(0, 100, 1) if seeds is None else seeds
    results = [
        single_fold_metrics(
            model_formula, dataset, split_ratio=split_ratio, seed=seed, thresh=thresh, scale_data=scale_data
        )
        for seed in range(len(seeds))
    ]
    return np.array(results)


def summarize_metrics(metric_data: np.ndarray) -> dict:
    """Median, std, and 95% CI of half-width for precision and recall."""
    precisions = [m["precision"] for m in metric_data]
    recalls = [m["recall"] for m in metric_data]

    def _stats(values):
        """Median, std, and 95% CI half-width of `values`."""
        return {
            "median": np.nanmedian(values),
            "std": np.nanstd(values),
            "ci95": 1.96 * np.std(values) / np.sqrt(len(values)),
        }

    p, r = _stats(precisions), _stats(recalls)
    return {
        "precision_median": p["median"], "precision_std": p["std"], "precision_ci95": p["ci95"],
        "recall_median": r["median"], "recall_std": r["std"], "recall_ci95": r["ci95"],
    }


