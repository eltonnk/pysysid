from dataclasses import dataclass, field
from typing import List

import numpy as np


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
