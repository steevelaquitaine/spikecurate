"""End-to-end smoke test on synthetic SpikeInterface data.

Uses a "perfect" sorter (the sorting output equals the ground truth) so
agreement scores and quality metrics are well-defined even on a small toy
recording, and exercises the full spikecurate.run() pipeline.
"""
import numpy as np
import pytest
import spikeinterface.full as si

import spikecurate


@pytest.fixture
def toy_data(tmp_path):
    recording, sorting_true = si.generate_ground_truth_recording(
        durations=[30],
        sampling_frequency=30000.0,
        num_channels=8,
        num_units=6,
        seed=0,
    )
    recording = recording.save(folder=tmp_path / "recording")
    sorting_true = sorting_true.save(folder=tmp_path / "gt")

    # perfect sorter: sorted output == ground truth
    sorting = sorting_true.select_units(unit_ids=sorting_true.unit_ids)

    single_unit_ids = sorting.unit_ids
    # arbitrary but valid partition into good/bad for the smoke test
    half = len(single_unit_ids) // 2
    good_unit_ids = single_unit_ids[:half]
    bad_unit_ids = single_unit_ids[half:]

    return {
        "recording": recording,
        "sorting_true": sorting_true,
        "sorting": sorting,
        "single_unit_ids": single_unit_ids,
        "good_unit_ids": good_unit_ids,
        "bad_unit_ids": bad_unit_ids,
        "we_save_path": str(tmp_path / "we_single_units"),
    }


def test_run_pipeline_end_to_end(toy_data):
    result = spikecurate.run(
        recording=toy_data["recording"],
        sorting_true=toy_data["sorting_true"],
        sorting=toy_data["sorting"],
        we_save_path=toy_data["we_save_path"],
        single_unit_ids=toy_data["single_unit_ids"],
        good_unit_ids=toy_data["good_unit_ids"],
        bad_unit_ids=toy_data["bad_unit_ids"],
        eval_seeds=np.arange(0, 5, 1),
        job_kwargs=dict(n_jobs=1, progress_bar=False),
    )

    dataset = result["data"]["dataset"]
    assert set(dataset.index) == set(toy_data["single_unit_ids"])
    assert "quality_label" in dataset.columns

    corr = result["feature_correlations"]
    assert list(corr.index) == list(corr.columns)

    stats = result["results"]["metric_stats"]
    assert 0 <= stats["precision_median"] <= 1
    assert 0 <= stats["recall_median"] <= 1
