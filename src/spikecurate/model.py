"""Fractional-logistic classifier for single-unit sorting quality.

Vendored and cleaned up from spikebias's src/nodes/models/Flc/models.py.
"""
import numpy as np
import pandas as pd

from . import crossval


class FractionalLogisticClassifier:
    """GLM (binomial family) classifier predicting `quality_label` (1=good,
    0=bad sorted single-unit) from a set of quality-metric predictors.
    """

    def __init__(self, predictors):
        self.predictors = list(predictors)
        self.formula = self._build_formula(self.predictors)

    @staticmethod
    def _build_formula(predictors) -> str:
        variables = "".join(f" + {p}" for p in predictors)
        return f"quality_label ~ 1{variables}"

    def evaluate(
        self,
        dataset: pd.DataFrame,
        seeds: np.ndarray = None,
        split_ratio: float = 0.75,
        thresh: float = 0.8,
        scale_data: bool = False,
    ) -> dict:
        """Cross-validated precision/recall over many random train/test splits."""
        seeds = np.arange(0, 100, 1) if seeds is None else seeds
        metric_data = crossval.crossval_metrics(
            dataset=dataset,
            model_formula=self.formula,
            split_ratio=split_ratio,
            seeds=seeds,
            thresh=thresh,
            scale_data=scale_data,
        )
        metric_stats = crossval.summarize_metrics(metric_data)
        return {"metric_stats": metric_stats, "metric_data": metric_data}

    def fit(
        self,
        dataset: pd.DataFrame,
        thresh: float = 0.8,
        scale_data: bool = False,
        regularization: str = "elastic_net",
        maxiter: int = 100,
        cnvrg_tol: float = 1e-10,
    ) -> dict:
        """Fit on the full dataset (with regularized warm-start) and self-evaluate."""
        return crossval.fit_on_full_dataset(
            self.formula,
            dataset,
            thresh=thresh,
            scale_data=scale_data,
            regularization=regularization,
            maxiter=maxiter,
            cnvrg_tol=cnvrg_tol,
        )
