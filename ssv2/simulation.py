import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm


class BaseSimulation:
    """Do simulation of bead dispersion.

    A key thing to understand is that the simulation is done in terms of
    x_pos and y_pos, which are integer coordinates of the beads in the hexagonal
    grid. The x_coord and y_coord are the actual euclidean coordinates of the
    beads, but we only compute them sometimes: for calculation of dispersion and
    for plotting.

    The reason we do this is because we can match on these x_pos and y_pos
    values without error. If we were to use the x_coord and y_coord values,
    we encounter difficulties with floating point error.

    This enables us to do a lot of computation once for dispersion in the middle
    of the grid, and then just move the dataframe to the focal bead's location,
    mask out things from the dataframe that are outside the grid, and then
    sample from a multinomial distribution.

    Parameters
    ----------
    row_count : int
        The number of rows in the hexagonal grid.
    beads_per_row : int
        The number of beads per row in the grid.
    max_dispersion_radius : int
        The maximum dispersion radius around a focal bead.
    max_dispersion_scale : float
        The scale to apply to the weight function of dispersion.
    joint_sums : pd.DataFrame
        A dataframe of joint sums of read counts, including columns `row_sums`
        and `col_sums`. This dataframe will serve as the joint empirical
        distribution of the read counts for those beads. It will be repeated if
        it's not sufficiently long to match the number of beads.
    bead_diameter : float, optional
        The physical diameter of a bead. Defaults to 1.
    bead_type_count : int, optional
        The number of distinct bead types, used for computing bead
        compabilitity. Defaults to 20.
    """

    def __init__(
        self,
        row_count,
        beads_per_row,
        max_dispersion_radius,
        max_dispersion_scale,
        joint_sums,
        bead_diameter=1,
        bead_type_count=20,
    ):
        self.bead_diameter = bead_diameter
        self.prob_beads_compatible = (bead_type_count - 1) / bead_type_count
        self.max_dispersion_radius = max_dispersion_radius
        self.max_dispersion_scale = max_dispersion_scale
        self.weight_fn = lambda r: self.max_dispersion_scale / (r**2) if r != 0 else 0.0
        self.bead_df = self.build_hexagonal_grid(row_count, beads_per_row)
        self.central_bead_idx = len(self.bead_df) // 2
        central_bead = self.bead_df.iloc[self.central_bead_idx]
        self.central_bead_x_pos = central_bead["x_pos"]
        self.central_bead_y_pos = central_bead["y_pos"]
        self.beads_near_center_df = self.build_beads_near_center_df(self.bead_df)
        self.max_x = np.max(self.bead_df["x_pos"])
        self.max_y = np.max(self.bead_df["y_pos"])

        # assign "row_sums" and "col_sums" as columns to self.bead_df, repeating the values as necessary to fill out the dataframe
        repetitions = len(self.bead_df) // len(joint_sums) + 1
        repeated_joint_sums = pd.concat([joint_sums] * repetitions, ignore_index=True)
        repeated_joint_sums = repeated_joint_sums.iloc[: len(self.bead_df)]
        self.bead_df["row_sums"] = repeated_joint_sums["row_sums"]
        self.bead_df["col_sums"] = repeated_joint_sums["col_sums"]
        self.indexed_bead_df = self.bead_df.set_index(["x_pos", "y_pos"])

    def build_hexagonal_grid(self, row_count, beads_per_row):
        """Build the basic hexagonal grid in terms of x and y positions."""
        grid_data = [
            (x_pos, y_pos)
            for y_pos in range(row_count)
            for x_pos in range(beads_per_row)
        ]
        df = pd.DataFrame(grid_data, columns=["x_pos", "y_pos"])
        return df

    def compute_coords_from_pos(self, row):
        """Compute the x and y coordinates from the x_pos and y_pos values."""
        delta_y = self.bead_diameter * np.sqrt(3) / 2
        y = row["y_pos"] * delta_y
        start_x = (self.bead_diameter / 2) * (row["y_pos"] % 2)
        x = start_x + row["x_pos"] * self.bead_diameter
        return x, y

    def add_coords(self, df_with_pos):
        """Add x and y coordinates to a dataframe with x_pos and y_pos columns."""
        df = df_with_pos.copy()
        df["x_coord"], df["y_coord"] = zip(
            *df.apply(self.compute_coords_from_pos, axis=1)
        )
        return df

    def plot_grid(self, df, ax, **kwargs):
        my_df = self.add_coords(df)
        ax.scatter(
            my_df["x_coord"], my_df["y_coord"], s=(self.bead_diameter**2) * 75, **kwargs
        )

    def plot_setting(self):
        """Plot the hexagonal grid with beads near the center along with their weight."""
        fig, ax = plt.subplots()
        self.plot_grid(self.bead_df, ax, edgecolors="gray", alpha=0.05)
        self.plot_grid(
            self.beads_near_center_df,
            ax,
            color=ListedColormap(["#e66101"]).colors[0],
            edgecolors="black",
            alpha=self.beads_near_center_df["weight"]
            / np.max(self.beads_near_center_df["weight"]),
        )
        self.plot_grid(
            self.beads_near_center_df,
            ax,
            color="none",
            edgecolors="gray",
        )
        ax.set_title("Hexagonal Grid Showing Weight Function from Central Bead")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        plt.show()

    def build_beads_near_center_df(self, bead_df):
        """This is a key aspect of the simulation. We build a dataframe of beads
        that are close to the center, and calculate their weights. We then use
        this dataframe repeatedly for other beads, masking to handle edges and
        then drawing from a multinomial distribution.
        """
        # Make a new copy and add coordinates.
        beads_near_center_df = self.add_coords(bead_df)

        central_bead_x_coord, central_bead_y_coord = self.compute_coords_from_pos(
            {"x_pos": self.central_bead_x_pos, "y_pos": self.central_bead_y_pos}
        )

        # Calculate distances from the center.
        beads_near_center_df["dist_from_center"] = np.sqrt(
            (beads_near_center_df["x_coord"] - central_bead_x_coord) ** 2
            + (beads_near_center_df["y_coord"] - central_bead_y_coord) ** 2
        )

        # Filter out beads that are too far from the center.
        beads_near_center_df = beads_near_center_df[
            beads_near_center_df["dist_from_center"] <= self.max_dispersion_radius
        ].copy()
        beads_near_center_df["weight"] = beads_near_center_df["dist_from_center"].apply(
            self.weight_fn
        )

        # Calculate the relative x and y positions from the center.
        beads_near_center_df["delta_x_pos"] = (
            beads_near_center_df["x_pos"] - self.central_bead_x_pos
        )
        beads_near_center_df["delta_y_pos"] = (
            beads_near_center_df["y_pos"] - self.central_bead_y_pos
        )
        return beads_near_center_df

    def simulate_bead_dispersion(self, focal_bead_idx):
        """This is the main routine for the simulation."""
        # Our strategy is to start with the dataframe of beads near the center,
        # and then move that dataframe to the focal bead's location, making
        # modifications as needed.
        sample_df = self.beads_near_center_df.copy()

        focal_bead_x_pos = self.bead_df.iloc[focal_bead_idx]["x_pos"]
        focal_bead_y_pos = self.bead_df.iloc[focal_bead_idx]["y_pos"]

        sample_df["x_pos"] = focal_bead_x_pos + sample_df["delta_x_pos"]
        sample_df["y_pos"] = focal_bead_y_pos + sample_df["delta_y_pos"]

        # Filter out beads that are outside the grid.
        sample_df = sample_df[
            (sample_df["x_pos"] >= 0)
            & (sample_df["x_pos"] <= self.max_x)
            & (sample_df["y_pos"] >= 0)
            & (sample_df["y_pos"] <= self.max_y)
        ]
        sample_df = sample_df.set_index(["x_pos", "y_pos"])
        # Note indexing magic. Because we are using the x_pos and y_pos as the
        # index in both data frames, assignment just works.
        sample_df["col_sums"] = self.indexed_bead_df["col_sums"]
        # Scale the weights by the column sums, which represent the natural
        # "attractiveness" of the beads.
        sample_df["weight"] = sample_df["weight"] * sample_df["col_sums"]
        sample_df.reset_index(inplace=True)
        read_count = self.indexed_bead_df.loc[focal_bead_x_pos, focal_bead_y_pos][
            "row_sums"
        ]

        # We will start with an unnormalized probability distribution
        # proportional to the weight values.
        bead_probs = sample_df["weight"].values
        # We then sample for which beads are compatible.
        bead_probs *= np.random.binomial(1, self.prob_beads_compatible, len(bead_probs))
        # Set the zero element to 0, because a bead cannot be compatible with itself.
        bead_probs[0] = 0.0
        # Normalize the probabilities.
        bead_probs /= np.sum(bead_probs)
        # Sample from a multinomial distribution.
        sample_df["bead_counts"] = np.random.multinomial(read_count, bead_probs)
        # Filter out beads with zero counts.
        sample_df = sample_df[sample_df["bead_counts"] > 0]
        sample_df = sample_df[["x_pos", "y_pos", "bead_counts"]]

        # Get the proper indices corresponding to the original bead_df.
        rows_pre_merge = len(sample_df)
        sample_df = sample_df.merge(
            self.bead_df.reset_index(), on=["x_pos", "y_pos"], how="inner"
        )[["x_pos", "y_pos", "bead_counts", "index"]]
        assert len(sample_df) == rows_pre_merge
        # We adopt the authoritative index of the beads as per the original bead_df.
        sample_df.set_index("index", inplace=True)
        return sample_df

    def plot_sample(self, focal_bead_idx):
        sample_df = self.add_coords(self.simulate_bead_dispersion(focal_bead_idx))
        bead_df = self.add_coords(self.bead_df)
        fig, ax = plt.subplots()
        self.plot_grid(bead_df, ax, edgecolors="gray", alpha=0.05)
        self.plot_grid(
            sample_df,
            ax,
            color=ListedColormap(["#e66101"]).colors[0],
            alpha=sample_df["bead_counts"] / np.max(sample_df["bead_counts"]),
        )

        # Add blue circle for the focal bead
        focal_x = bead_df.loc[focal_bead_idx]["x_coord"]
        focal_y = bead_df.loc[focal_bead_idx]["y_coord"]
        ax.scatter(focal_x, focal_y, color="blue", edgecolors=None, zorder=3)
        # s=(self.bead_diameter**2)*100,

        ax.set_title("Hexagonal Grid with Sampled Beads Highlighted")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        return fig

    def simulate_experiment(self):
        """Simulate an experiment with read counts drawn randomly from the read_counts np.array."""
        simulated_beads = []

        for source_bead in tqdm(range(len(self.bead_df))):
            df = self.simulate_bead_dispersion(source_bead)["bead_counts"]

            df = df.reset_index().rename(columns={df.index.name: "target_bead"})
            df["source_bead"] = source_bead
            simulated_beads.append(df)

        return pd.concat(simulated_beads)[["source_bead", "target_bead", "bead_counts"]]
