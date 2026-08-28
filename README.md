# spikecurate

Curate sorted single-unit quality from SpikeInterface extractors, using a
fractional-logistic classifier trained on SpikeInterface quality metrics.

Given a recording, a ground-truth sorting, a sorter's output, the sorter's
own single-unit calls, and expert good/bad labels for a subset of those
units, `spikecurate` engineers quality-metric features, trains a classifier
of accurately- vs. poorly-sorted single units, and reports its
cross-validated precision/recall.

## Install

```bash
pip install -e .
```

## Usage

```python
import spikecurate

result = spikecurate.run(
    recording=recording,          # spikeinterface RecordingExtractor
    sorting_true=sorting_true,    # spikeinterface ground-truth SortingExtractor
    sorting=sorting,              # spikeinterface SortingExtractor (e.g. Kilosort4)
    we_save_path="waveforms/single_units",
    single_unit_ids=single_unit_ids,   # e.g. sorting.unit_ids[sorting.get_property("KSLabel") == "good"]
    good_unit_ids=good_unit_ids,       # subset of single_unit_ids, >80% agreement with ground truth
    bad_unit_ids=bad_unit_ids,         # remaining subset of single_unit_ids
    job_kwargs=dict(n_jobs=-1, progress_bar=True),
)

result["feature_correlations"]           # pd.DataFrame: feature correlation matrix
result["results"]["metric_stats"]        # median/std/95% CI precision & recall
result["model"].formula                  # fitted GLM formula
```

Each pipeline stage is also usable standalone:

```python
from spikecurate.features import load_dataset
from spikecurate.model import FractionalLogisticClassifier
from spikecurate.plotting import plot_precision_recall
```

## Notes

- Requires a `principal_components` extension on the WaveformExtractor
  (computed automatically by `spikecurate.run`) - the silhouette quality
  metric depends on it.
- `good_unit_ids` and `bad_unit_ids` must partition `single_unit_ids`
  exactly (every single unit gets exactly one label) - `spikecurate.run`
  will raise `AssertionError` otherwise.
