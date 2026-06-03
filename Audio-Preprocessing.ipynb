import warnings
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torchaudio
import torch

from scipy.signal import butter, lfilter

warnings.filterwarnings("ignore")


def butter_bandpass(
    lowcut,
    highcut,
    fs,
    order=2
):
    nyq = 0.5 * fs

    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    return b, a


def butter_bandpass_filter(
    data,
    lowcut,
    highcut,
    fs,
    order=2
):
    b, a = butter_bandpass(
        lowcut,
        highcut,
        fs,
        order
    )

    return lfilter(
        b,
        a,
        data
    )


def normalize_audio(
    audio_data,
    target_range=(-1, 1)
):
    current_min = np.min(audio_data)
    current_max = np.max(audio_data)

    if current_max == current_min:
        return np.zeros_like(audio_data)

    normalized = (
        audio_data - current_min
    ) / (
        current_max - current_min
    )

    target_min, target_max = target_range

    normalized = (
        normalized *
        (target_max - target_min)
        + target_min
    )

    return normalized


def load_audio(
    file_path
):
    waveform, sr = torchaudio.load(
        file_path
    )

    audio = waveform.numpy()[0]

    return audio, waveform, sr


def resample_audio(
    waveform,
    original_sr,
    target_sr=2000
):
    if original_sr == target_sr:
        return waveform.numpy()[0]

    resampler = torchaudio.transforms.Resample(
        original_sr,
        target_sr
    )

    resampled = resampler(
        waveform
    )

    return resampled.numpy()[0]


def preprocess_audio(
    file_path,
    target_sr=2000,
    lowcut=0.5,
    highcut=500,
    filter_order=5
):
    original_audio, waveform, original_sr = load_audio(
        file_path
    )

    resampled_audio = resample_audio(
        waveform,
        original_sr,
        target_sr
    )

    normalized_audio = normalize_audio(
        resampled_audio
    )

    filtered_audio = butter_bandpass_filter(
        normalized_audio,
        lowcut,
        highcut,
        target_sr,
        filter_order
    )

    return (
        original_audio,
        filtered_audio,
        original_sr,
        target_sr
    )


def save_audio(
    audio,
    sr,
    output_path
):
    tensor = torch.tensor(
        audio
    ).unsqueeze(0)

    torchaudio.save(
        output_path,
        tensor,
        sr
    )


def plot_comparison(
    original_audio,
    filtered_audio,
    original_sr,
    filtered_sr,
    duration=3
):
    n1 = min(
        int(duration * original_sr),
        len(original_audio)
    )

    n2 = min(
        int(duration * filtered_sr),
        len(filtered_audio)
    )

    t1 = np.arange(n1) / original_sr
    t2 = np.arange(n2) / filtered_sr

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(14, 8)
    )

    ax[0].plot(
        t1,
        original_audio[:n1]
    )
    ax[0].set_title(
        "Original Signal"
    )

    ax[0].set_xlabel(
        "Time (s)"
    )

    ax[0].set_ylabel(
        "Amplitude"
    )

    ax[1].plot(
        t2,
        filtered_audio[:n2]
    )

    ax[1].set_title(
        "Filtered Signal"
    )

    ax[1].set_xlabel(
        "Time (s)"
    )

    ax[1].set_ylabel(
        "Amplitude"
    )

    plt.tight_layout()
    plt.show()


def frequency_analysis(
    audio,
    sr
):
    n = len(audio)

    fft = np.fft.fft(audio)
    freq = np.fft.fftfreq(
        n,
        1 / sr
    )

    mask = freq >= 0

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        freq[mask],
        np.abs(fft[mask])
    )

    plt.title(
        "Frequency Spectrum"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Magnitude"
    )

    plt.xlim(
        0,
        1000
    )

    plt.grid(True)

    plt.show()
