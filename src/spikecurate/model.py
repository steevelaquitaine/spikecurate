"""Fractional-logistic classifier for single-unit sorting quality.

API is intentionally scikit-learn-flavored: train/predict, with fit state
stored on the instance.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import StandardScaler

from . import crossval

LABEL_COL = crossval.LABEL_COL


class FractionalLogisticClassifier:
    """GLM (binomial family) classifier predicting `quality_label` (1=good,
    0=bad sorted single-unit) from a set of quality-metric predictors.

    Usage:
        model = FractionalLogisticClassifier(predictors)
        model.train(dataset)                  # fit on the full labeled dataset
        model.crossval_evaluate(dataset)       # cross-validated precision/recall
        model.predict(features)                # good/bad label for new units
    """

    def __init__(self, predictors, thresh: float = 0.8):
        """Args:
            predictors: quality-metric column names to use as GLM predictors.
            thresh: probability threshold above which a unit is predicted "good".
        """
        self.predictors = list(predictors)
        self.formula = self._build_formula(self.predictors)
        self.thresh = thresh

        # set by train()
        self.result_ = None
        self.scaler_ = None
        self.scale_data_ = False

    @staticmethod
    def _build_formula(predictors) -> str:
        """Build the `quality_label ~ 1 + p1 + p2 + ...` GLM formula string."""
        variables = "".join(f" + {p}" for p in predictors)
        return f"{LABEL_COL} ~ 1{variables}"

    @property
    def is_trained(self) -> bool:
        """Whether `train()` has been called."""
        return self.result_ is not None

    def train(
        self,
        dataset: pd.DataFrame,
        scale_data: bool = False,
        regularization: str = "elastic_net",
        maxiter: int = 100,
        cnvrg_tol: float = 1e-10,
    ) -> "FractionalLogisticClassifier":
        """Fit on the full labeled dataset (with a regularized warm-start,
        then one unregularized refit to recover p-values - see
        statsmodels' GLM.fit_regularized docs). Stores the fitted result
        (and, if `scale_data`, the fitted scaler) on the instance for
        later `predict()` calls.

        Returns self, so this chains: `model.train(dataset).predict(...)`.
        """
        assert LABEL_COL in dataset.columns, f"dataset must contain a '{LABEL_COL}' column"

        data = dataset.copy()
        self.scale_data_ = scale_data
        if scale_data:
            self.scaler_ = StandardScaler()
            data[self.predictors] = self.scaler_.fit_transform(data[self.predictors])
        else:
            self.scaler_ = None

        glm = sm.GLM.from_formula(self.formula, family=sm.families.Binomial(), data=data)
        try:
            warm_start = glm.fit_regularized(method=regularization, maxiter=maxiter, cnvrg_tol=cnvrg_tol)
            self.result_ = glm.fit(start_params=warm_start.params, maxiter=1)
        except Exception:
            # the regularized warm-start can fail to bracket a solution on
            # small/degenerate datasets (e.g. near-perfect separation) -
            # fall back to a plain fit, which crossval_evaluate's per-fold
            # fits already rely on successfully
            self.result_ = glm.fit()
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Classify units from their quality-metric features.

        Args:
            features: DataFrame indexed by unit id, with (at least) the
                columns in `self.predictors`. Any `quality_label` column is
                ignored - this is for units whose label you want predicted,
                labeled or not.

        Returns:
            DataFrame indexed like `features`, with columns:
                probability: P(good unit) from the fitted GLM
                predicted_label: 1 (good) / 0 (bad), thresholded at `self.thresh`
                predicted_quality: "good" / "bad"
        """
        if not self.is_trained:
            raise RuntimeError("call train() before predict()")

        x = features.loc[:, self.predictors].copy()
        if self.scale_data_:
            x[self.predictors] = self.scaler_.transform(x[self.predictors])

        # match the exact feature order/subset the fitted model uses
        ordered_features = self.result_.params.index[1:]
        x = x.loc[:, ordered_features]
        x.insert(0, "intercept", 1)

        probability = self.result_.predict(x)
        predicted_label = (probability >= self.thresh).astype(int)
        return pd.DataFrame(
            {
                "probability": probability,
                "predicted_label": predicted_label,
                "predicted_quality": np.where(predicted_label == 1, "good", "bad"),
            },
            index=features.index,
        )

    def crossval_evaluate(
        self,
        dataset: pd.DataFrame,
        seeds: np.ndarray = None,
        split_ratio: float = 0.75,
        scale_data: bool = False,
    ) -> dict:
        """Cross-validated precision/recall over many random train/test
        splits (one per seed). Does not require - or affect - `train()`;
        each fold fits its own model on a subset of `dataset`.
        """
        seeds = np.arange(0, 100, 1) if seeds is None else seeds
        metric_data = crossval.crossval_metrics(
            dataset=dataset,
            model_formula=self.formula,
            split_ratio=split_ratio,
            seeds=seeds,
            thresh=self.thresh,
            scale_data=scale_data,
        )
        metric_stats = crossval.summarize_metrics(metric_data)
        return {"metric_stats": metric_stats, "metric_data": metric_data}

    def score(self, dataset: pd.DataFrame) -> dict:
        """Precision/recall of the trained model against `dataset`'s own
        `quality_label` column (e.g. the training set, or a held-out set
        with known labels). Requires `train()` first.
        """
        predictions = self.predict(dataset)
        return {
            "precision": sk_metrics.precision_score(
                dataset[LABEL_COL], predictions["predicted_label"], zero_division=0
            ),
            "recall": sk_metrics.recall_score(
                dataset[LABEL_COL], predictions["predicted_label"], zero_division=0
            ),
        }
