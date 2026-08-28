# spikecurate

Curate sorted single-unit quality from SpikeInterface extractors, using a
fractional-logistic classifier trained on SpikeInterface quality metrics.

Given a recording, a ground-truth sorting, a sorter's output, the sorter's
own single-unit calls, and expert good/bad labels for a subset of those
units, `spikecurate` engineers quality-metric features, trains a classifier
of accurately- vs. poorly-sorted single units, and reports its
cross-validated precision/recall.

See the [GitHub repository](https://github.com/steevelaquitaine/spikecurate)
for the full README, demo notebook, and source.

## Install

```bash
pip install spikecurate
```

## Quickstart

```python
import spikecurate

result = spikecurate.run(
    recording=recording,          # spikeinterface RecordingExtractor
    sorting_true=sorting_true,    # spikeinterface ground-truth SortingExtractor
    sorting=sorting,              # spikeinterface SortingExtractor (e.g. Kilosort4)
    we_save_path="waveforms/single_units",
    single_unit_ids=single_unit_ids,
    good_unit_ids=good_unit_ids,
    bad_unit_ids=bad_unit_ids,
    predict_unit_ids=single_unit_ids[:10],
)

result["feature_correlations"]  # feature correlation matrix
result["results"]               # cross-validated precision/recall
result["model"]                 # trained FractionalLogisticClassifier
result["predictions"]           # probability, predicted_label, predicted_quality per unit
```

`spikecurate.run` is a convenience pipeline built on top of
{py:class}`~spikecurate.model.FractionalLogisticClassifier`, which is also
usable standalone - see the API reference below.

```{toctree}
:maxdepth: 2
:hidden:

api
```
