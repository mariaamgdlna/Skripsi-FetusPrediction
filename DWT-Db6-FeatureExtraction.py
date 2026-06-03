import pywt
import torch
import torchaudio

def wavelet_decompose(
    signal,
    wavelet="db6",
    max_levels=5
):
    approximations = {}
    details = {}

    current_signal = signal.copy()

    for level in range(
        1,
        max_levels + 1
    ):
        approx, detail = pywt.dwt(
            current_signal,
            wavelet,
            mode="symmetric"
        )

        approximations[level] = approx
        details[level] = detail

        current_signal = approx

    return (
        approximations,
        details
    )

def normalize_component(
    component
):
    if np.max(
        np.abs(component)
    ) > 0:

        return (
            component
            / np.max(
                np.abs(component)
            )
        )

    return component

def save_wavelet_component(
    component,
    sample_rate,
    output_path
):
    tensor = torch.tensor(
        component,
        dtype=torch.float32
    ).unsqueeze(0)

    torchaudio.save(
        output_path,
        tensor,
        sample_rate
    )

def save_approximation_levels(
    approximations,
    sample_rate,
    output_dir,
    base_name
):
    saved_files = []

    for level, signal in approximations.items():

        normalized_signal = (
            normalize_component(
                signal
            )
        )

        level_sr = (
            sample_rate
            // (2 ** level)
        )

        filename = (
            f"{base_name}_A{level}.wav"
        )

        filepath = (
            f"{output_dir}/{filename}"
        )

        save_wavelet_component(
            normalized_signal,
            level_sr,
            filepath
        )

        saved_files.append(
            filename
        )

    return saved_files

def save_detail_levels(
    details,
    sample_rate,
    output_dir,
    base_name
):
    saved_files = []

    for level, signal in details.items():

        normalized_signal = (
            normalize_component(
                signal
            )
        )

        level_sr = (
            sample_rate
            // (2 ** level)
        )

        filename = (
            f"{base_name}_D{level}.wav"
        )

        filepath = (
            f"{output_dir}/{filename}"
        )

        save_wavelet_component(
            normalized_signal,
            level_sr,
            filepath
        )

        saved_files.append(
            filename
        )

    return saved_files

def plot_wavelet_tree(
    signal,
    sample_rate,
    approximations,
    details
):
    max_levels = len(
        approximations
    )

    fig, axes = plt.subplots(
        (max_levels * 2) + 1,
        1,
        figsize=(15, 20)
    )

    time_original = np.linspace(
        0,
        len(signal) / sample_rate,
        len(signal)
    )

    axes[0].plot(
        time_original,
        signal
    )

    axes[0].set_title(
        "Original Signal"
    )

    axes[0].set_ylim(
        -1,
        1
    )

    for level in range(
        1,
        max_levels + 1
    ):
        approx_idx = (
            (level - 1) * 2 + 1
        )

        detail_idx = (
            (level - 1) * 2 + 2
        )

        approx_signal = (
            normalize_component(
                approximations[level]
            )
        )

        detail_signal = (
            normalize_component(
                details[level]
            )
        )

        approx_time = np.linspace(
            0,
            len(signal) / sample_rate,
            len(approx_signal)
        )

        detail_time = np.linspace(
            0,
            len(signal) / sample_rate,
            len(detail_signal)
        )

        axes[approx_idx].plot(
            approx_time,
            approx_signal
        )

        axes[approx_idx].set_title(
            f"A{level}"
        )

        axes[approx_idx].set_ylim(
            -1,
            1
        )

        axes[detail_idx].plot(
            detail_time,
            detail_signal
        )

        axes[detail_idx].set_title(
            f"D{level}"
        )

        axes[detail_idx].set_ylim(
            -1,
            1
        )

    plt.tight_layout()

    return fig

def extract_wavelet_coefficients(
    signal,
    wavelet="db6",
    max_levels=5
):
    approximations, details = (
        wavelet_decompose(
            signal,
            wavelet,
            max_levels
        )
    )

    return {
        "approximations":
            approximations,
        "details":
            details
    }
