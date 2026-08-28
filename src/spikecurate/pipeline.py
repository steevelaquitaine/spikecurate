"""End-to-end pipeline: extract single-unit waveforms, engineer quality-metric
features, and train/evaluate a classifier of sorted single-unit quality.
"""
import logging

import numpy as np
from spikeinterface import extract_waveforms
from spikeinterface.postprocessing import compute_principal_components

from .features import load_dataset
from .model import FractionalLogisticClassifier

logger = logging.getLogger(__name__)

DEFAULT_EXTRACT_WAVEFORMS_KWARGS = dict(
    mode="folder",
    sparse=True,
    ms_before=6,
    ms_after=6,
    max_spikes_per_unit=500,
    overwrite=True,
    seed=0,
)


def run(
    recording,
    sorting_true,
    sorting,
    we_save_path: str,
    single_unit_ids: np.ndarray,
    good_unit_ids: np.ndarray,
    bad_unit_ids: np.ndarray,
    *,
    predict_unit_ids: np.ndarray = None,
    delta_time: float = 1.3,
    n_components_pca: int = 5,
    load_we_if_exists: bool = False,
    load_qm_if_exists: bool = False,
    extract_waveforms_kwargs: dict = None,
    job_kwargs: dict = None,
    eval_seeds: np.ndarray = None,
    eval_split_ratio: float = 0.75,
    eval_thresh: float = 0.8,
    eval_scale_data: bool = False,
) -> dict:
    """Run the full single-unit quality curation pipeline.

    Args:
        recording: SpikeInterface RecordingExtractor.
        sorting_true: SpikeInterface ground-truth SortingExtractor.
        sorting: SpikeInterface SortingExtractor for the sorter under test
            (e.g. Kilosort4 output), containing all sorted units.
        we_save_path: folder to save (or load) the WaveformExtractor for the
            single units selected via `single_unit_ids`.
        single_unit_ids: unit ids in `sorting` classified as single units by
            the sorter (e.g. Kilosort4's KSLabel == "good").
        good_unit_ids: subset of `single_unit_ids` labeled accurately-sorted
            (>80% agreement with ground truth) - positive class for training.
        bad_unit_ids: subset of `single_unit_ids` labeled poorly-sorted -
            negative class for training. Every id in `single_unit_ids` must
            appear in exactly one of `good_unit_ids`/`bad_unit_ids`.
        predict_unit_ids: unit ids (subset of `single_unit_ids`) to classify
            with the model trained on the full dataset - e.g. a handful of
            example units to demo the classifier on. Defaults to all of
            `single_unit_ids`.
        delta_time: coincidence window (ms) used to compute sorted/ground-truth
            spike-train agreement scores.
        n_components_pca: number of PCA components computed per channel,
            required for the silhouette quality metric.
        load_we_if_exists / load_qm_if_exists: reuse a previously computed
            WaveformExtractor / quality-metrics extension at `we_save_path`
            instead of recomputing them.
        extract_waveforms_kwargs: overrides for
            `spikeinterface.extract_waveforms` (ms_before, ms_after,
            max_spikes_per_unit, sparse, ...).
        job_kwargs: SpikeInterface parallelization kwargs (n_jobs, ...).
        eval_seeds, eval_split_ratio, eval_thresh, eval_scale_data: passed to
            `FractionalLogisticClassifier.crossval_evaluate`/`train`.

    Returns:
        dict with keys:
            "we": the single-unit WaveformExtractor
            "data": output of `features.load_dataset` (dataset, predictors, ...)
            "feature_correlations": pd.DataFrame, dataset.corr()
            "model": the FractionalLogisticClassifier, trained on the full dataset
            "results": crossval_evaluate() output (metric_data, metric_stats)
            "predictions": model.predict() output for `predict_unit_ids`
    """
    job_kwargs = job_kwargs or {}
    we_kwargs = {**DEFAULT_EXTRACT_WAVEFORMS_KWARGS, **(extract_waveforms_kwargs or {})}

    # 1. select single units and (re)extract their waveforms
    sorting_single_units = sorting.select_units(unit_ids=list(single_unit_ids))
    if load_we_if_exists:
        from spikeinterface import WaveformExtractor
        we = WaveformExtractor.load_from_folder(we_save_path)
    else:
        we = extract_waveforms(recording, sorting_single_units, we_save_path, **we_kwargs, **job_kwargs)
        compute_principal_components(
            waveform_extractor=we, n_components=n_components_pca, mode="by_channel_local", **job_kwargs
        )

    # 2. engineer quality-metric features
    data = load_dataset(
        single_unit_ids,
        good_unit_ids,
        bad_unit_ids,
        sorting_single_units,
        sorting_true,
        we,
        load_qm_if_exists=load_qm_if_exists,
        delta_time=delta_time,
        job_kwargs=job_kwargs,
    )

    # 3. basic statistics on features
    feature_correlations = data["dataset"].corr()

    # 4. cross-validated performance (each fold trains its own model)
    model = FractionalLogisticClassifier(data["predictors"], thresh=eval_thresh)
    results = model.crossval_evaluate(
        data["dataset"],
        seeds=eval_seeds,
        split_ratio=eval_split_ratio,
        scale_data=eval_scale_data,
    )

    # 5. train the deployed model on the full dataset, then classify
    # predict_unit_ids with it
    model.train(data["dataset"], scale_data=eval_scale_data)

    predict_unit_ids = data["dataset"].index if predict_unit_ids is None else [int(u) for u in predict_unit_ids]
    predictions = model.predict(data["dataset"].loc[predict_unit_ids, model.predictors])

    return {
        "we": we,
        "data": data,
        "feature_correlations": feature_correlations,
        "model": model,
        "results": results,
        "predictions": predictions,
    }
