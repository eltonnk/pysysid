from dataclasses import dataclass, field
from typing import List

import numpy as np
import scipy.interpolate as sint


@dataclass
class SignalGenerator:
    offset: float = 0.0

    def value_at_t(self, t: float) -> float:
        raise NotImplementedError("Derived class should override this method")


class InputGenerator:
    def __init__(self, list_signal_generators: List[SignalGenerator]):
        self.list_sg = list_signal_generators
        self.input_size = len(self.list_sg)

        self.input_signal = np.zeros((self.input_size, 1))

    def value_at_t(self, t: float) -> np.ndarray:
        for i, sg in enumerate(self.list_sg):
            self.input_signal[i, 0] = sg.value_at_t(t)

        return self.input_signal

    def generate_timeseries(self, t_arr: np.ndarray) -> np.ndarray:
        # vect_value_at_t = np.vectorize(self.value_at_t, signature="()->(n)")

        # return vect_value_at_t(t_arr)
        n_samples = t_arr.shape[0]

        self.input_arr = np.zeros((self.input_size, n_samples))

        for i, t in enumerate(t_arr):
            self.input_arr[:, i] = self.value_at_t(t).reshape(self.input_size)

        return self.input_arr


@dataclass
class SquareGenerator(SignalGenerator):
    period: float = 1.0
    pulse_width: float = 0.5
    amplitude: float = 1.0

    def __post_init__(self):
        if self.pulse_width < 0 or self.pulse_width > 1:
            raise ValueError(
                f"""Attribute 'pulse_width' must have a must have a value in 
                the [0.0, 1.0] interval. Current value is {self.pulse_width}."""
            )
        self.half_period = self.period * self.pulse_width
        self.m_amplitude = -self.amplitude

    def value_at_t(self, t: float) -> float:
        t_remainder = t

        while t_remainder >= self.period:
            t_remainder -= self.period

        return (
            self.amplitude if t_remainder >= self.half_period else self.m_amplitude
        ) + self.offset


@dataclass
class StepGenerator(SignalGenerator):
    on_at: float = 0.0
    amplitude: float = 1.0

    def value_at_t(self, t: float) -> float:
        return (0 if t < self.on_at else self.amplitude) + self.offset


@dataclass
class SineGenerator(SignalGenerator):
    frequency: float = 1.0
    amplitude: float = 1.0
    phase: float = 0.0

    def value_at_t(self, t: float) -> float:
        return (
            self.amplitude * np.sin(2.0 * np.pi * self.frequency * t + self.phase)
            + self.offset
        )


@dataclass
class StairGenerator(SignalGenerator):
    period: float = 1.0
    step_amplitude: float = 1.0

    def value_at_t(self, t: float) -> float:
        t_remainder = t
        i = 0

        while t_remainder >= self.period:
            t_remainder -= self.period
            i = i + 1

        return self.step_amplitude * i + self.offset


@dataclass
class ChirpGenerator(SignalGenerator):
    start_frequency: float = 1.0
    end_frequency: float = 1.0
    chirp_length: float = 1.0  # In seconds
    amplitude: float = 1.0
    phase: float = 0.0

    def __post_init__(self):
        self.delta_frequency = self.end_frequency - self.start_frequency

    def value_at_t(self, t: float) -> float:
        instant_freq = (
            self.start_frequency + (self.delta_frequency) * t / self.chirp_length
        )
        return (
            self.amplitude * np.sin(2.0 * np.pi * instant_freq * t + self.phase)
            + self.offset
        )


class RepeatedChirpGenerator(ChirpGenerator):
    def value_at_t(self, t: float) -> float:
        t_remainder = t

        while t_remainder >= self.chirp_length:
            t_remainder -= self.chirp_length

        return super().value_at_t(t_remainder)


@dataclass
class PrbsGenerator(SignalGenerator):
    """Pseudo Random Binary Sequence Generator

    Thanks to Steven Dahdah for providing the original source of the
    code found in this class in https://github.com/decargroup/quanser_qube

    """

    amplitude: float = 1.0
    min_period: float = 1.0
    seed: int = 1.0

    def __post_init__(self):
        self.prbs_bits = self._prbs_bits()
        self.complete_bits_seq_len = len(self.prbs_bits)

    def _prbs_bits(self) -> np.ndarray:
        complete_sequence = []
        lfsr = self.seed & 0xFFFF
        while True:
            # Generate a new bit
            bit = (
                (
                    ((( ((lfsr >> 0) & 0xFFFF)
                    ^ ((lfsr >> 2) & 0xFFFF)) & 0xFFFF
                    ^ ((lfsr >> 3) & 0xFFFF)) & 0xFFFF
                    ^ ((lfsr >> 5) & 0xFFFF)) & 0xFFFF
                )
                & 0x0001
            ) & 0xFFFF
            # Shift new bit into register
            lfsr = (((lfsr >> 1) & 0xFFFF) | ((bit << 15) & 0xFFFF)) & 0xFFFF
            # Generate output boolean
            complete_sequence.append(bit == 0x0001)
            if lfsr == self.seed:
                break

        return np.array(complete_sequence)

    def value_at_t(self, t: float) -> float:
        t_remainder = t
        i = 0

        while t_remainder >= self.min_period:
            t_remainder -= self.min_period
            i = i + 1

        i = i % self.complete_bits_seq_len
        return (
            self.offset + self.amplitude
            if self.prbs_bits[i]
            else self.offset - self.amplitude
        )


def default_signal() -> np.ndarray:
    return np.zeros(
        10,
    )


@dataclass
class PredeterminedSignalGenerator(SignalGenerator):
    signal: np.ndarray = field(default_factory=default_signal)
    f_samp: float = 1.0

    def __post_init__(self):
        self.samp_period = 1.0 / self.f_samp

    def value_at_t(self, t: float) -> float:
        # Assuming zero-order hold discretized signal
        t_remainder = t
        i = 0

        while t_remainder >= self.samp_period:
            t_remainder -= self.samp_period
            i = i + 1

        return self.signal[i]


@dataclass
class InterpolatedSignalGenerator(PredeterminedSignalGenerator):
    time_arr: np.ndarray = field(default_factory=default_signal)
    kind: str = None

    def __post_init__(self):
        super().__post_init__()

        self.f_interp = sint.interp1d(self.time_arr, self.signal, kind=self.kind)

    def value_at_t(self, t: float) -> float:
        return self.f_interp(t)
