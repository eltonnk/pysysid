import numpy as np

import pysysid.signal_generators as sg


class TestInputGenerators:
    def test_generate_timeseries(self):
        signal_gen = sg.InputGenerator(
            [
                sg.SquareGenerator(period=1, pulse_width=0.5, amplitude=1),
                sg.SineGenerator(frequency=0.3, amplitude=0.01, phase=0),
            ]
        )

        t_arr = np.arange(0, 5)

        input_arr = signal_gen.generate_timeseries(t_arr)

        np.testing.assert_allclose(
            input_arr,
            np.array(
                [
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [0.0, 0.00951057, -0.00587785, -0.00587785, 0.00951057],
                ]
            ),
            rtol=0.0001,
        )
