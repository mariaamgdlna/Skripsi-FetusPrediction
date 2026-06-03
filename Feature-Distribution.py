import numpy as np
import pandas as pd


def histogram_interval_analysis(
        feature_values,
        n_bins=10):
    """
    Histogram-based interval analysis.

    Parameters
    ----------
    feature_values : array-like
        Feature values (e.g. Average_Peak_Value)

    n_bins : int
        Number of intervals

    Returns
    -------
    summary_df : DataFrame
        Interval statistics
    """

    feature_values = np.array(feature_values)

    min_val = np.min(feature_values)
    max_val = np.max(feature_values)

    bin_width = (max_val - min_val) / n_bins

    interval_results = []

    for i in range(n_bins):

        bin_start = min_val + (i * bin_width)
        bin_end = min_val + ((i + 1) * bin_width)

        if i == n_bins - 1:
            mask = (
                (feature_values >= bin_start)
                &
                (feature_values <= bin_end)
            )
        else:
            mask = (
                (feature_values >= bin_start)
                &
                (feature_values < bin_end)
            )

        count = np.sum(mask)

        percentage = (
            count / len(feature_values)
        ) * 100

        interval_results.append({
            "interval_start":
                bin_start,

            "interval_end":
                bin_end,

            "count":
                count,

            "percentage":
                percentage
        })

    summary_df = pd.DataFrame(
        interval_results
    )

    return summary_df


def identify_distribution_pattern(
        feature_values,
        n_bins=10):
    """
    Identify dominant interval.
    """

    summary_df = histogram_interval_analysis(
        feature_values,
        n_bins
    )

    dominant_interval = summary_df.loc[
        summary_df["count"].idxmax()
    ]

    distribution_result = {
        "total_samples":
            len(feature_values),

        "minimum":
            np.min(feature_values),

        "maximum":
            np.max(feature_values),

        "mean":
            np.mean(feature_values),

        "std":
            np.std(feature_values),

        "dominant_interval_start":
            dominant_interval["interval_start"],

        "dominant_interval_end":
            dominant_interval["interval_end"],

        "dominant_frequency":
            dominant_interval["count"],

        "dominant_percentage":
            dominant_interval["percentage"]
    }

    return distribution_result


def feature_distribution_analysis(
        feature_values,
        n_bins=10):
    """
    Complete pipeline:

    Feature Values
            ↓
    Histogram Binning
            ↓
    Interval Analysis
            ↓
    Distribution Statistics
    """

    interval_table = histogram_interval_analysis(
        feature_values,
        n_bins
    )

    distribution_summary = (
        identify_distribution_pattern(
            feature_values,
            n_bins
        )
    )

    return (
        interval_table,
        distribution_summary
    )
