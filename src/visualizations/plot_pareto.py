import sys

sys.path.append("src")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from base import initialize_session, System
import random
from collections import OrderedDict


def compute_pareto_frontier(df, maximize_x=True, maximize_y=True):
    """
    Computes the Pareto frontier for a given DataFrame.

    Parameters:
    - df: pandas DataFrame with 'x' and 'y' columns.
    - maximize_x: Boolean, whether to maximize 'x'.
    - maximize_y: Boolean, whether to maximize 'y'.

    Returns:
    - pandas DataFrame containing the Pareto frontier.
    """
    # Sort the data based on 'x' and 'y'
    df_sorted = df.sort_values(
        by=["x", "y"], ascending=(not maximize_x, not maximize_y)
    ).reset_index(drop=True)

    # Initialize the Pareto frontier
    pareto_front = []
    current_max = -np.inf if maximize_y else np.inf

    for index, row in df_sorted.iterrows():
        y = row["y"]
        if (maximize_y and y > current_max) or (not maximize_y and y < current_max):
            pareto_front.append(row)
            current_max = y

    return pd.DataFrame(pareto_front)


def plot_pareto_frontiers(systems):
    """
    Plots Pareto frontiers (median safety vs median capability) and all points for each generation.

    Parameters:
    - systems: List of System objects containing the necessary attributes.
    """
    # Step 1: Extract Relevant Data from Systems
    data = []
    for system in systems:
        if system.system_capability_ci_median == 0:
            continue  # Skip systems with median capability of 0
        data.append(
            {
                "generation_timestamp": system.generation_timestamp,
                "median_capability": system.system_capability_ci_median,
                "median_safety": system.system_safety_ci_median,
            }
        )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        print("No data available to plot.")
        return

    # Step 2: Group by Generation
    generations = df.groupby("generation_timestamp")

    # Define color map
    cmap = plt.get_cmap("tab10")
    colors = cmap.colors

    plt.figure(figsize=(14, 10))

    # Step 3: Compute and Plot Pareto Frontiers and Scatter Points for Each Generation
    for i, (gen, group) in enumerate(generations):
        # Assign a color to the generation
        color = colors[i % len(colors)]

        # Rename columns for clarity
        group = group.rename(columns={"median_capability": "x", "median_safety": "y"})

        # Plot all points in the generation as scatter
        plt.scatter(
            group["x"],
            group["y"],
            color=color,
            alpha=0.5,
            label=f"Generation {gen} - Systems",
            edgecolor="k",
            s=50,
        )

        # Compute Pareto frontier
        pareto_df = compute_pareto_frontier(group, maximize_x=True, maximize_y=True)

        # Sort Pareto frontier by 'x' to ensure lines are drawn correctly
        pareto_df = pareto_df.sort_values(by="x")

        # Plot Pareto frontier as straight lines between points
        plt.plot(
            pareto_df["x"],
            pareto_df["y"],
            marker="o",
            linestyle="-",
            label=f"Generation {gen} - Pareto Frontier",
            color=color,
            linewidth=2,
        )

    # Customize Plot
    plt.xlabel("Median Capability", fontsize=14)
    plt.ylabel("Median Safety", fontsize=14)
    plt.title("Pareto Frontiers and Systems by Generation", fontsize=16)
    plt.legend(loc="best", fontsize=10, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    random.seed(42)
    population_id = "445b50db-5b72-4399-b078-f4dba75e2109"

    all_systems = []
    for session in initialize_session():
        systems = session.query(System).filter_by(population_id=population_id).all()
        all_systems.extend(systems)

        plot_pareto_frontiers(all_systems)
