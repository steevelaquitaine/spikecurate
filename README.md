<p align="center">
  <img src="https://raw.githubusercontent.com/steevelaquitaine/spikecurate/master/asset/logo_spikecurate.png" alt="spikecurate logo" width="200">
</p>

# spikecurate

author: steeve laquitaine - laquitainesteeve@gmail.com

[![tests](https://github.com/steevelaquitaine/spikecurate/actions/workflows/tests.yml/badge.svg)](https://github.com/steevelaquitaine/spikecurate/actions/workflows/tests.yml)
[![regression](https://github.com/steevelaquitaine/spikecurate/actions/workflows/regression.yml/badge.svg)](https://github.com/steevelaquitaine/spikecurate/actions/workflows/regression.yml)
[![docs](https://github.com/steevelaquitaine/spikecurate/actions/workflows/docs.yml/badge.svg)](https://steevelaquitaine.github.io/spikecurate/)

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

## Documentation

Full API reference (built with Sphinx from the docstrings in `src/spikecurate/`):
**https://steevelaquitaine.github.io/spikecurate/**

Rebuilds automatically on every push to `master`
(`.github/workflows/docs.yml`). To build it locally:

```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

## Demo notebook

`demo/demo_03.ipynb` walks through `FractionalLogisticClassifier`'s API -
`train`, `crossval_evaluate`, `predict`, `score` - using the real,
already-engineered feature dataset checked into this repo at
`dataset/single_unit_quality_dataset.csv`. No SpikeInterface extractors
needed, just `pandas` + `spikecurate`:

```bash
pip install -e .
python -m ipykernel install --user --name spikecurate --display-name spikecurate
jupyter notebook demo/demo_03.ipynb
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
  `dataset/single_unit_quality_dataset.csv` - the same real engineered
  dataset `demo_03.ipynb` walks through - so a future change to the
  model/crossval/predict code that silently changes results, not just one
  that crashes, gets caught. Needs no SpikeInterface extractors, so it's
  fast. Runs on every push to `master`
  (`.github/workflows/regression.yml`).

## Citation

If you use this package, please consider citing:

> Laquitaine, S., Imbeni, M., Tharayil, J., Isbister, J. B., & Reimann, M. W. (2024).
> Spike sorting biases and information loss in a detailed cortical model. *bioRxiv*.
> https://doi.org/10.1101/2024.12.04.626805

```bibtex
@article{laquitaine2024spikesorting,
  title   = {Spike sorting biases and information loss in a detailed cortical model},
  author  = {Laquitaine, Steeve and Imbeni, Milo and Tharayil, Joseph and Isbister, James B. and Reimann, Michael W.},
  journal = {bioRxiv},
  year    = {2024},
  doi     = {10.1101/2024.12.04.626805},
  url     = {https://www.biorxiv.org/content/10.1101/2024.12.04.626805v1}
}
```

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
  (`dataset.sort_index()`). `FractionalLogisticClassifier.crossval_evaluate`'s
  cross-validation splits train/test by row *position*
  (`random.sample(range(n), n_train)`), so for a given `dataset` and
  `seeds`, results are reproducible only if row order is pinned - passing
  an unsorted dataset (e.g. one you built yourself, in whatever order the
  sorting extractor's unit ids happened to come in) will still run, but
  won't reproduce the same split as the same content sorted differently.
