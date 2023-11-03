import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


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
        self.central_bead_x = central_bead["x_coord"]
        self.central_bead_y = central_bead["y_coord"]
        self.beads_near_center_df = self.build_beads_near_center_df(self.bead_df)
        self.max_x = np.max(self.bead_df["x_coord"])
        self.max_y = np.max(self.bead_df["y_coord"])

    def round_coordinates(self, df):
        df["x_coord"] = df["x_coord"].apply(lambda x: round(x, 7))
        df["y_coord"] = df["y_coord"].apply(lambda x: round(x, 7))
        return df

    def build_hexagonal_grid(self, row_count, beads_per_row):
        coords = []

        # Vertical spacing between rows
        delta_y = self.bead_diameter * np.sqrt(3) / 2

        for i in range(row_count):
            y = i * delta_y

            # Every other row is shifted
            start_x = (self.bead_diameter / 2) * (i % 2)

            for j in range(beads_per_row):
                x = start_x + j * self.bead_diameter
                coords.append((x, y))

        df = pd.DataFrame(coords, columns=["x_coord", "y_coord"])
        df = self.round_coordinates(df)
        return df

    def plot_grid(self, df, ax, **kwargs):
        ax.scatter(
            df["x_coord"], df["y_coord"], s=(self.bead_diameter**2) * 75, **kwargs
        )

    def plot_setting(self):
        fig, ax = plt.subplots()
        self.plot_grid(self.bead_df, ax, edgecolors="gray", alpha=0.05)
        self.plot_grid(
            self.beads_near_center_df,
            ax,
            color=ListedColormap(["#e66101"]).colors[0],
            alpha=self.beads_near_center_df["weight"]
            / np.max(self.beads_near_center_df["weight"]),
        )
        ax.set_title("Hexagonal Grid Showing weight Function from Central Bead")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        plt.show()

    def build_beads_near_center_df(self, bead_df):
        # First we build a bounding rectangle around the central bead to avoid
        # calculating distances for all beads
        x_min = self.central_bead_x - self.max_dispersion_radius
        x_max = self.central_bead_x + self.max_dispersion_radius
        y_min = self.central_bead_y - self.max_dispersion_radius
        y_max = self.central_bead_y + self.max_dispersion_radius

        # Filter beads within bounding rectangle
        bounded_df = bead_df[
            (bead_df["x_coord"] >= x_min)
            & (bead_df["x_coord"] <= x_max)
            & (bead_df["y_coord"] >= y_min)
            & (bead_df["y_coord"] <= y_max)
        ].copy()

        # Calculate distances for the beads within bounding rectangle
        bounded_df["dist_from_center"] = np.sqrt(
            (bounded_df["x_coord"] - self.central_bead_x) ** 2
            + (bounded_df["y_coord"] - self.central_bead_y) ** 2
        )

        # Final filter based on exact distance
        beads_near_center_df = bounded_df[
            bounded_df["dist_from_center"] <= self.max_dispersion_radius
        ].copy()
        beads_near_center_df["weight"] = beads_near_center_df["dist_from_center"].apply(
            self.weight_fn
        )
        beads_near_center_df["x_displacement"] = (
            beads_near_center_df["x_coord"] - self.central_bead_x
        )
        beads_near_center_df["y_displacement"] = (
            beads_near_center_df["y_coord"] - self.central_bead_y
        )
        beads_near_center_df = self.round_coordinates(beads_near_center_df)
        return beads_near_center_df

    def simulate_bead_dispersion(self, read_count, focal_bead_idx):
        # Our strategy is to start with the dataframe of beads near the center,
        # and then move that dataframe to the focal bead's location, making
        # modifications as needed.
        sample_df = self.beads_near_center_df.copy()

        focal_bead_x = self.bead_df.iloc[focal_bead_idx]["x_coord"]
        focal_bead_y = self.bead_df.iloc[focal_bead_idx]["y_coord"]

        sample_df["x_coord"] = focal_bead_x + sample_df["x_displacement"]
        sample_df["y_coord"] = focal_bead_y + sample_df["y_displacement"]

        # Filter out beads that are outside the grid.
        sample_df = sample_df[
            (sample_df["x_coord"] >= 0)
            & (sample_df["x_coord"] <= self.max_x)
            & (sample_df["y_coord"] >= 0)
            & (sample_df["y_coord"] <= self.max_y)
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
        sample_df = sample_df[["x_coord", "y_coord", "bead_counts"]]
        sample_df = self.round_coordinates(sample_df)

        # Get the proper indices corresponding to the original bead_df.
        sample_df.index -= self.central_bead_idx
        assert min(sample_df.index) >= 0
        return sample_df

    def plot_sample(self, read_count, focal_bead_idx):
        sample_df = self.simulate_bead_dispersion(read_count, focal_bead_idx)
        fig, ax = plt.subplots()
        self.plot_grid(self.bead_df, ax, edgecolors="gray", alpha=0.05)
        self.plot_grid(
            sample_df,
            ax,
            color=ListedColormap(["#e66101"]).colors[0],
            alpha=sample_df["bead_counts"] / np.max(sample_df["bead_counts"]),
        )
        ax.set_title("Hexagonal Grid with Sampled Beads Highlighted")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        plt.show()
        print(sample_df)
