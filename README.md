# spikecurate

[![tests](https://github.com/steevelaquitaine/spikecurate/actions/workflows/tests.yml/badge.svg)](https://github.com/steevelaquitaine/spikecurate/actions/workflows/tests.yml)
[![regression](https://github.com/steevelaquitaine/spikecurate/actions/workflows/regression.yml/badge.svg)](https://github.com/steevelaquitaine/spikecurate/actions/workflows/regression.yml)

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

## Demo notebooks

- `demo/demo_01.ipynb` runs the full pipeline (`spikecurate.run`) against
  the real dataset produced by `spikebias`'s
  `notebooks/2_results_editorial_requests/fig5e.ipynb`.
- `demo/demo_02.ipynb` is a focused walkthrough of the classifier's own API
  underneath `run` - `train`, `crossval_evaluate`, `predict`, `score` -
  reusing the same pre-computed waveform extractor to skip straight to a
  `dataset` of engineered features.
- `demo/demo_03.ipynb` runs the same walkthrough as `demo_02.ipynb`, but
  starting from `dataset/single_unit_quality_dataset.csv` - the engineered
  feature dataset `demo_02.ipynb` produces, saved once and checked into
  this repo. No SpikeInterface extractors, no real `spikebias` dataset
  needed - just `pandas` + `spikecurate`, so it runs in seconds anywhere.

`demo_01.ipynb` and `demo_02.ipynb` need the real `spikebias` dataset,
which isn't bundled with this repo. To run them, use the same conda/mamba
environment `spikebias` itself uses, which pins the exact package versions
(notably `spikeinterface==0.100.8`) `demo_01.ipynb`'s recorded fidelity
check depends on:

```bash
# create the env (same spec as spikebias's envs/spikebias.yml)
mamba env create -f envs/spikebias.yml --prefix ./envs/spikebias
mamba activate ./envs/spikebias

# install spikecurate itself into it
pip install -e .

# register it as a Jupyter kernel and launch
python -m ipykernel install --user --name spikebias --display-name spikebias
jupyter notebook demo/demo_01.ipynb  # or demo/demo_02.ipynb
```

Update `SPIKEBIAS_DATASET_DIR` near the top of each notebook if your
`spikebias` checkout isn't at `/home/steeve/steeve/epfl/code/spikebias`.

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
    predict_unit_ids=single_unit_ids[:10],  # units to classify with the trained model; defaults to all
    job_kwargs=dict(n_jobs=-1, progress_bar=True),
)

result["feature_correlations"]           # pd.DataFrame: feature correlation matrix
result["results"]["metric_stats"]        # median/std/95% CI precision & recall, cross-validated
result["model"]                          # FractionalLogisticClassifier, trained on the full dataset
result["predictions"]                    # DataFrame indexed by unit id: probability, predicted_label, predicted_quality
```

### The classifier's own API

`spikecurate.run` is a convenience pipeline built on top of
`FractionalLogisticClassifier`, which has a scikit-learn-flavored API and is
usable standalone (e.g. to retrain/re-predict without redoing waveform
extraction, or to classify units engineered outside `spikecurate.run`):

```python
from spikecurate.model import FractionalLogisticClassifier

model = FractionalLogisticClassifier(predictors)   # predictors: result["data"]["predictors"]
model.crossval_evaluate(dataset)                    # cross-validated precision/recall (each fold trains its own fit)
model.train(dataset)                                 # fit on the full labeled dataset; returns self
model.predict(features)                              # DataFrame: probability, predicted_label, predicted_quality
model.score(dataset)                                 # precision/recall of the trained model against dataset's own labels
```

`features` passed to `predict` just needs unit id as its index and the
columns in `model.predictors` - it doesn't need a `quality_label` column
(that's the whole point: predicting labels for units you don't already
have labels for).

Other pipeline stages are also usable standalone:

```python
from spikecurate.features import load_dataset
from spikecurate.plotting import plot_precision_recall
```

## Testing

- `tests/test_pipeline.py`: end-to-end smoke test of `spikecurate.run()`
  against synthetic SpikeInterface data. Runs on every push/PR
  (`.github/workflows/tests.yml`).
- `tests/test_classification.py`: pins the classifier's numeric output
  (`crossval_evaluate`, `train`+`score`, `predict`) against
  `dataset/single_unit_quality_dataset.csv` - the real engineered dataset
  from `demo_02.ipynb`/`demo_03.ipynb` - so a future change to the
  model/crossval/predict code that silently changes results, not just one
  that crashes, gets caught. Needs no SpikeInterface extractors, so it's
  fast. Runs on every push to `master`
  (`.github/workflows/regression.yml`).

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE).

## Notes

- Requires a `principal_components` extension on the WaveformExtractor
  (computed automatically by `spikecurate.run`) - the silhouette quality
  metric depends on it.
- `good_unit_ids` and `bad_unit_ids` must partition `single_unit_ids`
  exactly (every single unit gets exactly one label) - `spikecurate.run`
  will raise `AssertionError` otherwise.
- `features.load_dataset` always returns `dataset` sorted by unit id
  (`dataset.sort_index()`). `FractionalLogisticClassifier.evaluate`'s
  cross-validation splits train/test by row *position*
  (`random.sample(range(n), n_train)`), so for a given `dataset` and
  `seeds`, results are reproducible only if row order is pinned - passing
  an unsorted dataset (e.g. one you built yourself, in whatever order the
  sorting extractor's unit ids happened to come in) will still run, but
  won't reproduce the same split as the same content sorted differently.
