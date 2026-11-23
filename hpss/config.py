from __future__ import annotations

from dataclasses import dataclass, field

SHARED_POLICIES = (
    "complementary",
    "split",
    "shared",
    "residual",
    "harmonic",
    "percussive",
)
MASK_KINDS = ("soft", "hard")


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int = 2048
    hop_length: int | None = None
    win_length: int | None = None
    window: str = "hann"
    center: bool = True
    pad_mode: str = "reflect"

    def __post_init__(self) -> None:
        if self.n_fft < 8:
            raise ValueError("n_fft must be >= 8")
        hop = self.n_fft // 4 if self.hop_length is None else self.hop_length
        if hop <= 0:
            raise ValueError("hop_length must be positive")
        object.__setattr__(self, "hop_length", hop)
        if self.win_length is None:
            object.__setattr__(self, "win_length", self.n_fft)
        if not (0 < self.win_length <= self.n_fft):
            raise ValueError("win_length must be in (0, n_fft]")

    @property
    def hop(self) -> int:
        return int(self.hop_length)


@dataclass(frozen=True)
class HPSSConfig:
    stft: STFTConfig = field(default_factory=STFTConfig)
    kernel_time: int = 31
    kernel_freq: int = 31
    mask_kind: str = "soft"
    mask_power: float = 2.0
    ambiguity_floor_db: float = -40.0
    shared_policy: str = "residual"
    duplicate_weight: float = 1.0
    energy_floor_db: float = -80.0

    def __post_init__(self) -> None:
        if self.kernel_time < 1 or self.kernel_freq < 1:
            raise ValueError("median kernel sizes must be >= 1")
        if self.mask_kind not in MASK_KINDS:
            raise ValueError(f"mask_kind must be one of {MASK_KINDS}")
        if self.mask_power <= 0:
            raise ValueError("mask_power must be positive")
        if self.ambiguity_floor_db >= 0:
            raise ValueError("ambiguity_floor_db must be negative")
        if self.shared_policy not in SHARED_POLICIES:
            raise ValueError(f"shared_policy must be one of {SHARED_POLICIES}")
        if self.duplicate_weight < 0:
            raise ValueError("duplicate_weight must be >= 0")
        if self.energy_floor_db >= 0:
            raise ValueError("energy_floor_db must be negative")

    def to_dict(self) -> dict:
        return {
            "n_fft": self.stft.n_fft,
            "hop_length": self.stft.hop,
            "win_length": self.stft.win_length,
            "window": self.stft.window,
            "center": self.stft.center,
            "pad_mode": self.stft.pad_mode,
            "kernel_time": self.kernel_time,
            "kernel_freq": self.kernel_freq,
            "mask_kind": self.mask_kind,
            "mask_power": self.mask_power,
            "ambiguity_floor_db": self.ambiguity_floor_db,
            "shared_policy": self.shared_policy,
            "duplicate_weight": self.duplicate_weight,
            "energy_floor_db": self.energy_floor_db,
        }
