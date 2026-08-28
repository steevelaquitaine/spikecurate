"""Plotting helpers for classifier evaluation results."""
import numpy as np
import pandas as pd
import seaborn as sns


def plot_precision_recall(ax, metric_data):
    """Strip plot of per-fold precision/recall with median +/- 95% CI."""
    precisions = [m["precision"] for m in metric_data]
    recalls = [m["recall"] for m in metric_data]

    df = pd.DataFrame(data=[precisions, recalls], index=["precision", "recall"]).T
    sns.stripplot(ax=ax, data=df, jitter=0.04, color="k", size=3)

    for x, values in enumerate([precisions, recalls]):
        ax.errorbar(
            x=x,
            y=np.nanmedian(values),
            yerr=1.96 * np.std(values) / np.sqrt(len(values)),
            marker="o",
            color="orange",
            markeredgecolor="w",
            markersize=5,
            zorder=np.inf,
        )

    ax.spines[["right", "top"]].set_visible(False)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    ax.set_ylim([0, 1])
    return ax
