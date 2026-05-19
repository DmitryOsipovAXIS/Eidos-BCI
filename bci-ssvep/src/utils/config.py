"""Central configuration for SSVEP frequencies, paths, and board defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SSVEPConfig:
    """Experiment parameters and filesystem layout.

    Attributes:
        left_hz: Flicker frequency mapped to LEFT / class 0.
        right_hz: Flicker frequency mapped to RIGHT / class 1.
        bandpass_margin_hz: Half-width around each target for simple bandpass.
        window_seconds: Length of each analysis window for FFT / classifier.
        project_root: Root of the ``bci-ssvep`` project (parent of ``src``).
    """

    left_hz: float = 8.0
    right_hz: float = 15.0
    bandpass_margin_hz: float = 2.0
    window_seconds: float = 4.0
    sample_rate_hz: float = 125.0
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    @property
    def data_raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "data" / "processed" / "models"

    def bandpass_band_hz(self) -> tuple[float, float]:
        """Wide band covering both SSVEP targets (for simple preprocessing)."""
        low = min(self.left_hz, self.right_hz) - self.bandpass_margin_hz
        high = max(self.left_hz, self.right_hz) + self.bandpass_margin_hz
        return max(low, 0.5), high

    def class_label_for_frequency(self, hz: float) -> int:
        """Return 0 (LEFT) or 1 (RIGHT) for the closer target frequency."""
        if abs(hz - self.left_hz) <= abs(hz - self.right_hz):
            return 0
        return 1
