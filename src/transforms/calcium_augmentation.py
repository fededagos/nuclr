"""Optional perturbations of calcium training views, after the legacy unit mask.

Only the training transform config instantiates this class. Validation and export
tokenize the untouched data directly. Never install it on a shared dataset.
"""

import numpy as np

from .unit_dropout import UnitDropout


class CalciumTrainingViews(UnitDropout):
    def __init__(
        self,
        min_units=0.5,
        augmentation="off",
        jitter_frames=1,
        gain_range=(0.8, 1.2),
        noise_sigma=0.2,
    ):
        super().__init__(min_units)
        if augmentation not in {"off", "jitter", "scale", "noise", "all"}:
            raise ValueError(f"Unknown augmentation: {augmentation}")
        if (
            isinstance(jitter_frames, bool)
            or not isinstance(jitter_frames, int)
            or jitter_frames < 0
        ):
            raise ValueError("jitter_frames must be a nonnegative integer")
        self.augmentation = augmentation
        self.jitter_frames = jitter_frames
        if (
            len(gain_range) != 2
            or not np.isfinite(gain_range).all()
            or not 0 < gain_range[0] <= gain_range[1]
        ):
            raise ValueError("gain_range must be finite, positive and ordered")
        self.gain_range = tuple(gain_range)
        if not np.isfinite(noise_sigma) or noise_sigma < 0:
            raise ValueError(
                "noise_sigma must be finite and nonnegative (baseline SD units)"
            )
        self.noise_sigma = float(noise_sigma)

    def __call__(self, data):
        data = super().__call__(data)
        if self.augmentation == "off":
            return data  # No allocation or random draw: exact legacy RNG stream.
        traces = data.calcium_traces
        frames, units = traces.df_over_f_norm.shape
        if not frames or not units:
            return data
        if self.augmentation in {"jitter", "all"}:
            shift = np.random.randint(
                -self.jitter_frames, self.jitter_frames + 1, size=units
            )
            # Constant-edge padding: never wrap the last response into the beginning.
            indices = np.clip(np.arange(frames)[:, None] - shift, 0, frames - 1)
            for key in ("df_over_f", "df_over_f_norm"):
                values = getattr(traces, key)
                setattr(traces, key, np.take_along_axis(values, indices, axis=0))
        if self.augmentation in {"scale", "all"}:
            gain = np.random.uniform(*self.gain_range, size=units)
            for key in ("df_over_f", "df_over_f_norm"):
                values = getattr(traces, key)
                setattr(traces, key, (values * gain).astype(values.dtype))
        if self.augmentation in {"noise", "all"}:
            raw, norm = traces.df_over_f, traces.df_over_f_norm
            # The exporter divides by ONE per-file p99. Recover that conversion
            # before adding noise: noise_sigma is in baseline SD, not p99 units.
            valid = np.isfinite(raw) & np.isfinite(norm) & (np.abs(raw) > 1e-6)
            if not valid.any():
                raise ValueError("Cannot infer baseline-SD normalization for noise")
            ratio = float(np.median(norm[valid] / raw[valid]))
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError("Expected positive per-file calcium normalization")
            noise = np.random.normal(0, self.noise_sigma, size=(frames, units))
            traces.df_over_f = (raw + noise).astype(raw.dtype)
            traces.df_over_f_norm = (norm + noise * ratio).astype(norm.dtype)
        return data
