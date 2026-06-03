import numpy as np
from scipy.signal import find_peaks


def convert_to_seconds(time_str):
    """
    Convert time format:
    1:04.34 -> 64.34 seconds
    """
    parts = time_str.split(':')

    if len(parts) == 1:
        return float(parts[0])

    minutes = int(parts[0])
    seconds = float(parts[1])

    return (minutes * 60) + seconds


def calculate_adaptive_threshold(signal, radius=10):
    """
    Adaptive threshold =
    mean(all peaks above zero)
    """

    peak_indices, _ = find_peaks(
        signal,
        distance=radius,
        height=0
    )

    if len(peak_indices) == 0:
        return 0

    peak_values = signal[peak_indices]

    peaks_above_zero = peak_values[
        peak_values > 0
    ]

    if len(peaks_above_zero) == 0:
        return 0

    threshold = np.mean(
        peaks_above_zero
    )

    return threshold


def sliding_window_peak_analysis(
        signal,
        fs,
        adaptive_threshold,
        window_duration=0.265,
        radius=10):
    """
    Sliding window analysis.

    Returns:
    --------
    features : list of dict
    """

    window_size = int(
        window_duration * fs
    )

    features = []

    start = 0
    segment_id = 1

    while start + window_size <= len(signal):

        end = start + window_size

        segment = signal[start:end]

        peak_indices, _ = find_peaks(
            segment,
            distance=radius,
            height=0
        )

        peak_values = segment[peak_indices]

        peaks_above_threshold = peak_values[
            peak_values >= adaptive_threshold
        ]

        peaks_below_threshold = peak_values[
            peak_values < adaptive_threshold
        ]

        feature = {
            "segment_id":
                segment_id,

            "start_time":
                start / fs,

            "end_time":
                end / fs,

            "total_peaks":
                len(peak_values),

            "peaks_above_threshold":
                len(peaks_above_threshold),

            "peaks_below_threshold":
                len(peaks_below_threshold),

            "avg_peak_value":
                np.mean(peaks_above_threshold)
                if len(peaks_above_threshold) > 0
                else 0,

            "std_peak_value":
                np.std(peaks_above_threshold)
                if len(peaks_above_threshold) > 0
                else 0,

            "max_peak_value":
                np.max(peaks_above_threshold)
                if len(peaks_above_threshold) > 0
                else 0,

            "min_peak_value":
                np.min(peaks_above_threshold)
                if len(peaks_above_threshold) > 0
                else 0,

            "peak_ratio":
                len(peaks_above_threshold)
                / len(peak_values)
                if len(peak_values) > 0
                else 0,

            "valid_segment":
                1 if len(peaks_above_threshold) > 0
                else 0
        }

        features.append(feature)

        start += window_size
        segment_id += 1

    return features


def extract_sliding_window_features(
        signal,
        fs,
        window_duration=0.265,
        radius=10):
    """
    Complete pipeline:
    Signal
        ↓
    Peak Detection
        ↓
    Adaptive Threshold
        ↓
    Sliding Window
        ↓
    Feature Extraction
    """

    adaptive_threshold = calculate_adaptive_threshold(
        signal,
        radius
    )

    features = sliding_window_peak_analysis(
        signal,
        fs,
        adaptive_threshold,
        window_duration,
        radius
    )

    return features
