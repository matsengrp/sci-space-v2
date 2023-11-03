import numpy as np

from ssv2.simulation import BaseSimulation


def test_run():
    simulator = BaseSimulation(29, 29)
    sample_df = simulator.simulate_bead_dispersion(20, 130)
