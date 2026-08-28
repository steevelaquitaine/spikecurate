"""Regression tests against dataset/single_unit_quality_dataset.csv.

This dataset is the engineered feature dataset produced (and explained) in
demo_02.ipynb / demo_03.ipynb, checked into the repo so these tests don't
need SpikeInterface extractors or the real spikebias dataset. They pin the
classifier's numeric output on this fixed dataset, so a future change to
the model/crossval/predict code that silently changes results - not just
one that crashes - gets caught.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spikecurate.model import FractionalLogisticClassifier

DATASET_PATH = Path(__file__).parent.parent / "dataset" / "single_unit_quality_dataset.csv"


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(DATASET_PATH, index_col="unit_id")


@pytest.fixture(scope="module")
def predictors(dataset):
    return [c for c in dataset.columns if c != "quality_label"]


@pytest.fixture(scope="module")
def trained_model(dataset, predictors):
    model = FractionalLogisticClassifier(predictors, thresh=0.8)
    model.train(dataset)
    return model


def test_crossval_evaluate_matches_demo_02(dataset, predictors):
    model = FractionalLogisticClassifier(predictors, thresh=0.8)
    stats = model.crossval_evaluate(dataset, seeds=np.arange(0, 100, 1))["metric_stats"]

    assert stats["precision_median"] == pytest.approx(0.8888888888888888)
    assert stats["recall_median"] == pytest.approx(0.5909090909090909)


def test_score_matches_demo_02(dataset, trained_model):
    score = trained_model.score(dataset)

    assert score["precision"] == pytest.approx(0.9803921568627451)
    assert score["recall"] == pytest.approx(0.5952380952380952)


def test_predict_matches_demo_02(dataset, predictors, trained_model):
    example_ids = list(dataset.index[:5])  # [26, 29, 40, 41, 42]
    predictions = trained_model.predict(dataset.loc[example_ids, predictors])

    assert list(predictions["predicted_label"]) == [0, 0, 0, 0, 1]
    assert list(predictions["predicted_quality"]) == ["bad", "bad", "bad", "bad", "good"]
    assert predictions.loc[40, "probability"] == pytest.approx(0.006578, abs=1e-6)
    assert predictions.loc[42, "probability"] == pytest.approx(0.913419, abs=1e-6)


def test_predict_before_train_raises(predictors):
    with pytest.raises(RuntimeError):
        FractionalLogisticClassifier(predictors).predict(pd.DataFrame())
