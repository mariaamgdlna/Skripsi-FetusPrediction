import os
import glob
import torchaudio
from datetime import datetime

def find_local_peaks(signal, radius=10):
    """
    Local peak detection
    """
    if np.min(signal) == np.max(signal):
        return np.array([], dtype=int)

    peak_indices = []

    for i in range(radius, len(signal) - radius):
        if signal[i] == np.max(signal[i-radius:i+radius+1]):
            peak_indices.append(i)

    return np.array(peak_indices)

def calculate_adaptive_threshold(signal, fs, radius=10):
    """
    Adaptive threshold using first 3 windows
    """

    duration = len(signal) / fs

    windows = [
        (0, 3),
        (3, 6),
        (6, 9)
    ]

    all_peaks_above_zero = []

    for start_sec, end_sec in windows:

        if start_sec >= duration:
            continue

        end_sec = min(end_sec, duration)

        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)

        segment = signal[start_idx:end_idx]

        peaks = find_local_peaks(segment, radius)

        if len(peaks) == 0:
            continue

        peak_values = segment[peaks]

        peak_values = peak_values[
            peak_values > 0
        ]

        all_peaks_above_zero.extend(
            peak_values.tolist()
        )

    if len(all_peaks_above_zero) == 0:
        return 0

    threshold = np.mean(
        all_peaks_above_zero
    )

    return threshold

def extract_peak_features(signal, fs, radius=10):
    """
    Peak-based feature extraction
    """

    threshold = calculate_adaptive_threshold(
        signal,
        fs,
        radius
    )

    peaks = find_local_peaks(
        signal,
        radius
    )

    if len(peaks) == 0:

        return {
            "adaptive_threshold": threshold,
            "peak_count": 0,
            "peak_above_threshold": 0,
            "peak_ratio": 0,
            "mean_peak_height": 0,
            "std_peak_height": 0,
            "max_peak_height": 0,
            "min_peak_height": 0,
            "mean_peak_interval": 0,
            "std_peak_interval": 0
        }

    peak_values = signal[peaks]

    peaks_above = peak_values[
        peak_values >= threshold
    ]

    peak_times = peaks / fs

    intervals = np.diff(
        peak_times
    )

    features = {

        "adaptive_threshold":
            threshold,

        "peak_count":
            len(peaks),

        "peak_above_threshold":
            len(peaks_above),

        "peak_ratio":
            len(peaks_above) / len(peaks),

        "mean_peak_height":
            np.mean(peak_values),

        "std_peak_height":
            np.std(peak_values),

        "max_peak_height":
            np.max(peak_values),

        "min_peak_height":
            np.min(peak_values),

        "mean_peak_interval":
            np.mean(intervals)
            if len(intervals) > 0 else 0,

        "std_peak_interval":
            np.std(intervals)
            if len(intervals) > 0 else 0
    }

    return features

def process_audio_file(audio_path, radius=10):
    """
    Extract features from one WAV file
    """

    waveform, fs = torchaudio.load(
        audio_path
    )

    signal = waveform.numpy()[0]

    features = extract_peak_features(
        signal,
        fs,
        radius
    )

    features["filename"] = os.path.basename(
        audio_path
    )

    return features

def process_folder(input_folder,
                   output_csv,
                   radius=10):
    """
    Process all WAV files
    """

    wav_files = glob.glob(
        os.path.join(
            input_folder,
            "*.wav"
        )
    )

    if len(wav_files) == 0:
        print("No WAV files found.")
        return

    all_features = []

    print(f"Found {len(wav_files)} files")

    for i, file_path in enumerate(wav_files):

        print(
            f"[{i+1}/{len(wav_files)}] "
            f"{os.path.basename(file_path)}"
        )

        try:

            features = process_audio_file(
                file_path,
                radius
            )

            all_features.append(
                features
            )

        except Exception as e:

            print(
                f"Error processing "
                f"{file_path}: {e}"
            )

    df = pd.DataFrame(
        all_features
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print()
    print("Feature extraction completed")
    print(f"Files processed : {len(df)}")
    print(f"Output saved    : {output_csv}")

    return df

def main():

    input_folder = (
        "/content/mydrive/MyDrive/"
        "SKENARIO 1 EMD DATA ORIGINAL/"
        "dwt_decomposition_tree"
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_csv = (
        "/content/mydrive/MyDrive/"
        f"peak_features_{timestamp}.csv"
    )

    process_folder(
        input_folder=input_folder,
        output_csv=output_csv,
        radius=10
    )


if __name__ == "__main__":
    main()
