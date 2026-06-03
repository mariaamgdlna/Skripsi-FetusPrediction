import numpy as np

from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline


class EMDDenoiser:

    def __init__(
        self,
        max_imfs=10,
        sift_threshold=0.2,
        max_sifts=50
    ):
        self.max_imfs = max_imfs
        self.sift_threshold = sift_threshold
        self.max_sifts = max_sifts

    def find_extrema(
        self,
        signal
    ):
        maxima_idx, _ = find_peaks(
            signal,
            distance=5
        )

        minima_idx, _ = find_peaks(
            -signal,
            distance=5
        )

        return maxima_idx, minima_idx

    def compute_envelopes(
        self,
        signal,
        maxima_idx,
        minima_idx
    ):
        t = np.arange(
            len(signal)
        )

        if (
            len(maxima_idx) < 2
            or len(minima_idx) < 2
        ):
            return (
                np.zeros_like(signal),
                np.zeros_like(signal)
            )

        if maxima_idx[0] != 0:
            maxima_idx = np.concatenate(
                [[0], maxima_idx]
            )

        if maxima_idx[-1] != len(signal) - 1:
            maxima_idx = np.concatenate(
                [maxima_idx, [len(signal) - 1]]
            )

        if minima_idx[0] != 0:
            minima_idx = np.concatenate(
                [[0], minima_idx]
            )

        if minima_idx[-1] != len(signal) - 1:
            minima_idx = np.concatenate(
                [minima_idx, [len(signal) - 1]]
            )

        upper = CubicSpline(
            maxima_idx,
            signal[maxima_idx]
        )(t)

        lower = CubicSpline(
            minima_idx,
            signal[minima_idx]
        )(t)

        return upper, lower

    def compute_mean_envelope(
        self,
        upper,
        lower
    ):
        return (
            upper + lower
        ) / 2

    def stopping_criterion(
        self,
        previous,
        current
    ):
        epsilon = 1e-10

        numerator = np.sum(
            (previous - current) ** 2
        )

        denominator = (
            np.sum(previous ** 2)
            + epsilon
        )

        return numerator / denominator

    def extract_imf(
        self,
        residual
    ):
        h = residual.copy()

        for _ in range(
            self.max_sifts
        ):
            previous = h.copy()

            maxima_idx, minima_idx = (
                self.find_extrema(h)
            )

            if (
                len(maxima_idx)
                + len(minima_idx)
                < 4
            ):
                break

            upper, lower = (
                self.compute_envelopes(
                    h,
                    maxima_idx,
                    minima_idx
                )
            )

            mean_env = (
                self.compute_mean_envelope(
                    upper,
                    lower
                )
            )

            h = h - mean_env

            sd = self.stopping_criterion(
                previous,
                h
            )

            if sd < self.sift_threshold:
                break

        return h

    def emd_decomposition(
        self,
        signal
    ):
        residual = signal.copy()

        imfs = []

        for _ in range(
            self.max_imfs
        ):
            imf = self.extract_imf(
                residual
            )

            imfs.append(imf)

            residual = (
                residual - imf
            )

            maxima_idx, minima_idx = (
                self.find_extrema(
                    residual
                )
            )

            if (
                len(maxima_idx)
                + len(minima_idx)
                < 2
            ):
                break

        imfs.append(
            residual
        )

        return np.array(imfs)

    def soft_threshold(
        self,
        signal,
        threshold
    ):
        return (
            np.sign(signal)
            * np.maximum(
                np.abs(signal)
                - threshold,
                0
            )
        )

    def estimate_noise_threshold(
        self,
        imfs
    ):
        first_imf = imfs[0]

        mad = np.median(
            np.abs(
                first_imf
                - np.median(first_imf)
            )
        )

        return (
            3 * mad / 0.6745
        )

    def denoise(
        self,
        signal
    ):
        imfs = self.emd_decomposition(
            signal
        )

        threshold = (
            self.estimate_noise_threshold(
                imfs
            )
        )

        denoised_imfs = []

        for i, imf in enumerate(
            imfs[:-1]
        ):
            if i < 3:

                adaptive_threshold = (
                    threshold
                    * (0.8 ** i)
                )

                denoised_imf = (
                    self.soft_threshold(
                        imf,
                        adaptive_threshold
                    )
                )

                denoised_imfs.append(
                    denoised_imf
                )

            else:
                denoised_imfs.append(
                    imf
                )

        denoised_imfs.append(
            imfs[-1]
        )

        denoised_signal = np.sum(
            denoised_imfs,
            axis=0
        )

        return (
            denoised_signal,
            imfs,
            denoised_imfs
        )
