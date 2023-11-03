import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm


class BaseSimulation:
    def __init__(
        self,
        row_count,
        beads_per_row,
        bead_diameter=1,
        bead_type_count=20,
        max_dispersion_radius=10,
    ):
        self.bead_diameter = bead_diameter
        self.prob_beads_compatible = (bead_type_count - 1) / bead_type_count
        self.max_dispersion_radius = max_dispersion_radius
        # TODO-- this is a hack
        self.max_dispersion_scale = max_dispersion_radius / 2
        self.weight_fn = (
            lambda r: self.max_dispersion_scale / (r**2) if r != 0 else 0.0
        )
        self.bead_df = self.build_hexagonal_grid(row_count, beads_per_row)
        self.central_bead_idx = len(self.bead_df) // 2
        central_bead = self.bead_df.iloc[self.central_bead_idx]
        self.central_bead_x_pos = central_bead["x_pos"]
        self.central_bead_y_pos = central_bead["y_pos"]
        self.beads_near_center_df = self.build_beads_near_center_df(self.bead_df)
        self.max_x = np.max(self.bead_df["x_pos"])
        self.max_y = np.max(self.bead_df["y_pos"])

    def build_hexagonal_grid(self, row_count, beads_per_row):
        grid_data = [
            (x_pos, y_pos)
            for y_pos in range(row_count)
            for x_pos in range(beads_per_row)
        ]
        df = pd.DataFrame(grid_data, columns=["x_pos", "y_pos"])
        return df

    def compute_coords_from_pos(self, row):
        delta_y = self.bead_diameter * np.sqrt(3) / 2
        y = row["y_pos"] * delta_y
        start_x = (self.bead_diameter / 2) * (row["y_pos"] % 2)
        x = start_x + row["x_pos"] * self.bead_diameter
        return x, y

    def add_coords(self, df_with_pos):
        df = df_with_pos.copy()
        df["x_coord"], df["y_coord"] = zip(
            *df.apply(self.compute_coords_from_pos, axis=1)
        )
        return df

    def plot_grid(self, df, ax, **kwargs):
        my_df = self.add_coords(df)
        ax.scatter(
            my_df["x_coord"],
            my_df["y_coord"],
            s=(self.bead_diameter**2) * 75,
            **kwargs
        )

    def plot_setting(self):
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
        beads_near_center_df = self.add_coords(bead_df)

        central_bead_x_coord, central_bead_y_coord = self.compute_coords_from_pos(
            {"x_pos": self.central_bead_x_pos, "y_pos": self.central_bead_y_pos}
        )

        # Calculate distances for the beads within bounding rectangle
        beads_near_center_df["dist_from_center"] = np.sqrt(
            (beads_near_center_df["x_coord"] - central_bead_x_coord) ** 2
            + (beads_near_center_df["y_coord"] - central_bead_y_coord) ** 2
        )

        beads_near_center_df = beads_near_center_df[
            beads_near_center_df["dist_from_center"] <= self.max_dispersion_radius
        ].copy()
        beads_near_center_df["weight"] = beads_near_center_df["dist_from_center"].apply(
            self.weight_fn
        )
        beads_near_center_df["delta_x_pos"] = (
            beads_near_center_df["x_pos"] - self.central_bead_x_pos
        )
        beads_near_center_df["delta_y_pos"] = (
            beads_near_center_df["y_pos"] - self.central_bead_y_pos
        )
        return beads_near_center_df

    def simulate_bead_dispersion(self, read_count, focal_bead_idx):
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
        sample_df.set_index("index", inplace=True)
        return sample_df

    def plot_sample(self, read_count, focal_bead_idx):
        sample_df = self.add_coords(
            self.simulate_bead_dispersion(read_count, focal_bead_idx)
        )
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

    def simulate_experiment(self, read_counts):
        simulated_beads = []

        for source_bead in tqdm(range(len(self.bead_df))):
            read_count = np.random.choice(read_counts)
            df = self.simulate_bead_dispersion(read_count, source_bead)["bead_counts"]

            df = df.reset_index().rename(columns={df.index.name: "target_bead"})
            df["source_bead"] = source_bead
            simulated_beads.append(df)

        return pd.concat(simulated_beads)[["source_bead", "target_bead", "bead_counts"]]
