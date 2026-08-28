"""Quality-metric feature engineering for sorted single units.

Vendored and cleaned up from spikebias's src/nodes/models/Flc/dataloader.py
and src/nodes/metrics/quality.get_scores, decoupled from that repo's
config/logging setup so this package has no dependency on it.
"""
import copy
import logging

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.core.template_tools as ttools
from spikeinterface import comparison
from spikeinterface.qualitymetrics import compute_quality_metrics as qm

logger = logging.getLogger(__name__)

SELECTED_QUALITY_METRICS = [
    "amplitude_cutoff",
    "firing_range",
    "firing_rate",
    "isi_violations_ratio",
    "presence_ratio",
    "rp_contamination",
    "rp_violations",
    "sd_ratio",
    "snr",
]


def get_agreement_scores(sorting_true, sorting, delta_time: float = 1.3) -> pd.DataFrame:
    """Agreement scores between ground-truth and sorted units."""
    comp = comparison.compare_sorter_to_ground_truth(
        sorting_true,
        sorting,
        exhaustive_gt=True,
        delta_time=delta_time,
    )
    return comp.agreement_scores


def _mad(data: np.ndarray) -> float:
    mean_data = np.mean(data)
    return np.mean(np.absolute(data - mean_data))


def _mad_ratio(spike_amp: np.ndarray, noise_amp: np.ndarray) -> float:
    """sd_ratio-like metric, robust to outliers."""
    return _mad(spike_amp) / _mad(noise_amp)


def _best_site_mad_noise(we, max_chids, unit):
    wv, _ = we.get_waveforms(unit_id=unit, with_index=True)
    c_ids = we.sparsity.unit_id_to_channel_ids[unit]
    max_chid = max_chids[unit]
    max_chid_ix = np.where(c_ids == max_chid)[0][0]
    return wv[:, :, max_chid_ix].flatten()


def _mad_ratio_all_units(unit_ids, we, spike_amp) -> list:
    max_chids = ttools.get_template_extremum_channel(we, peak_sign="both")
    return [
        _mad_ratio(spike_amp[unit], _best_site_mad_noise(we, max_chids, unit))
        for unit in unit_ids
    ]


def _add_spike_amplitude_extension(we, load_if_exists: bool, job_kwargs: dict):
    n_sites = we.get_num_channels()
    we.recording.set_property("gain_to_uV", np.ones((n_sites,)))
    we.recording.set_property("offset_to_uV", np.zeros((n_sites,)))

    if not load_if_exists:
        si.postprocessing.compute_spike_amplitudes(we, outputs="by_unit", **job_kwargs)
    else:
        we.load_extension("spike_amplitudes")

    assert we.has_extension("spike_amplitudes"), "failed to load spike_amplitudes extension"
    return we


def compute_quality_metrics(
    we,
    load_qm_if_exists: bool = False,
    skip_pc_metrics: bool = True,
    job_kwargs: dict = None,
) -> pd.DataFrame:
    """Compute SpikeInterface quality metrics for every unit in `we`.

    `we` must already have a `principal_components` extension computed
    (required for the silhouette metric) - see
    `spikeinterface.postprocessing.compute_principal_components`.
    """
    job_kwargs = job_kwargs or {}

    we = _add_spike_amplitude_extension(we, load_if_exists=load_qm_if_exists, job_kwargs=job_kwargs)

    qmetrics = qm(
        we,
        qm_params={
            "amplitude_cutoff": {
                "peak_sign": "neg",
                "num_histogram_bins": 100,
                "histogram_smoothing_value": 3,
                "amplitudes_bins_min_ratio": 0,
            }
        },
        load_if_exists=load_qm_if_exists,
        skip_pc_metrics=skip_pc_metrics,
        **job_kwargs,
    )
    qmetrics = qmetrics[SELECTED_QUALITY_METRICS]

    assert we.has_extension("principal_components"), (
        "run compute_principal_components(waveform_extractor=we, ...) before "
        "compute_quality_metrics() - required for the silhouette metric"
    )
    silhouette = qm(we, metric_names=["silhouette"], skip_pc_metrics=False, **job_kwargs)
    qmetrics["silhouette"] = silhouette.values

    logger.info("Quality metric completion: %s", qmetrics.notna().sum().to_dict())
    return qmetrics


def engineer_quality_features(
    single_unit_ids: np.ndarray,
    we,
    load_qm_if_exists: bool = False,
    job_kwargs: dict = None,
) -> pd.DataFrame:
    """Quality-metric feature matrix (rows=units, cols=metrics) for `single_unit_ids`."""
    job_kwargs = job_kwargs or {}

    qmetrics = compute_quality_metrics(we, load_qm_if_exists=load_qm_if_exists, job_kwargs=job_kwargs)

    spike_amp = si.postprocessing.compute_spike_amplitudes(
        we, peak_sign="neg", outputs="by_unit", load_if_exists=load_qm_if_exists, **job_kwargs
    )[0]
    qmetrics["mad_ratio"] = _mad_ratio_all_units(qmetrics.index, we, spike_amp)

    # spikeinterface's unit_ids come back as strings (e.g. Kilosort4 output);
    # normalize both sides to int so callers can pass single_unit_ids as
    # either an int array (from a CSV) or the sorting extractor's own
    # (string) unit_ids array
    qmetrics.index = qmetrics.index.astype(int)
    single_unit_ids = np.asarray([int(u) for u in single_unit_ids])

    missing = set(single_unit_ids) - set(qmetrics.index)
    if missing:
        raise ValueError(f"{len(missing)} single_unit_ids missing from computed quality metrics: {sorted(missing)[:10]}")

    return qmetrics.loc[list(single_unit_ids), :]


def build_labeled_dataset(
    qmetrics: pd.DataFrame,
    good_unit_ids: np.ndarray,
    bad_unit_ids: np.ndarray,
    sorting,
    sorting_true,
    delta_time: float = 1.3,
) -> tuple[pd.DataFrame, pd.Index]:
    """Attach ground-truth agreement scores are not used as a feature here;
    build the (features, quality_label) dataset used to train the classifier.

    quality_label: 1 for units in good_unit_ids, 0 for units in bad_unit_ids.
    """
    scores = get_agreement_scores(sorting_true, sorting, delta_time=delta_time)
    scores.columns = scores.columns.astype(int)

    missing = set(qmetrics.index) - set(scores.columns)
    if missing:
        raise ValueError(f"{len(missing)} unit ids missing from agreement scores: {sorted(missing)[:10]}")

    dataset = copy.copy(qmetrics)

    # presence_ratio is constant across units in the reference pipeline;
    # firing_rate is collinear with firing_range and sd_ratio with mad_ratio
    dataset = dataset.drop(columns=["presence_ratio"], errors="ignore")
    predictors = dataset.columns

    dataset["quality_label"] = np.nan
    dataset.loc[list(good_unit_ids), "quality_label"] = 1
    dataset.loc[list(bad_unit_ids), "quality_label"] = 0

    assert not np.isnan(dataset["quality_label"]).any(), (
        "some units are in neither good_unit_ids nor bad_unit_ids - "
        "every unit in qmetrics.index must be labeled"
    )

    return dataset, predictors


def load_dataset(
    single_unit_ids: np.ndarray,
    good_unit_ids: np.ndarray,
    bad_unit_ids: np.ndarray,
    sorting,
    sorting_true,
    we,
    load_qm_if_exists: bool = False,
    delta_time: float = 1.3,
    job_kwargs: dict = None,
) -> dict:
    """Engineer the full quality-metric feature dataset used to train/evaluate
    the single-unit quality classifier.

    Returns a dict with keys: dataset, predictors, qmetrics.
    """
    job_kwargs = job_kwargs or {}

    qmetrics = engineer_quality_features(
        single_unit_ids, we, load_qm_if_exists=load_qm_if_exists, job_kwargs=job_kwargs
    )
    dataset, predictors = build_labeled_dataset(
        qmetrics, good_unit_ids, bad_unit_ids, sorting, sorting_true, delta_time=delta_time
    )

    # drop units with a non-finite feature value
    bad_rows = dataset.index[np.isinf(dataset).any(axis=1)]
    if len(bad_rows):
        logger.info("dropping %d/%d units with a non-finite feature value", len(bad_rows), len(dataset))
    dataset = dataset.drop(index=bad_rows)

    # pin row order to unit id, independent of the order single_unit_ids
    # happened to arrive in: crossval.crossval_metrics splits by row
    # *position*, so an unpinned row order would make results depend on
    # upstream extractor/id ordering despite a fixed seed - sorting here
    # makes a given (dataset content, seed) pair reproduce the same split
    # regardless of how single_unit_ids was ordered by the caller
    dataset = dataset.sort_index()

    return {
        "dataset": dataset,
        "predictors": predictors,
        "qmetrics": qmetrics,
        "single_unit_ids": single_unit_ids,
        "good_unit_ids": good_unit_ids,
        "bad_unit_ids": bad_unit_ids,
    }
