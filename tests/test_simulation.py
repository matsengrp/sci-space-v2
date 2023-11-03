import numpy as np

from ssv2.simulation import BaseSimulation


def test_index_subtraction():
    """We do this tricky thing where we subtract out the central bead index
    from the index of the beads near the center. This test ensures that
    this is working as expected.
    """
    simulator = BaseSimulation(29, 29)
    sample_df = simulator.simulate_bead_dispersion(20, 0)

    merge_df = sample_df.merge(
        simulator.bead_df.reset_index(), on=["x_coord", "y_coord"], how="inner"
    )
    # This index is the index as judged by this merge. It should be the same
    # as the index calculated using sample_df, which was calculated using
    # this tricky index subtraction.
    assert np.all(merge_df["index"] == sample_df.index)
