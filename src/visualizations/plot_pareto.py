import sys

sys.path.append("src")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from base import initialize_session, System, Population
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
        if (
            not system.system_capability_ci_median
            or system.system_capability_ci_median == 0
        ):
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

    # Define a colorblind-friendly colormap with 11 discrete colors
    cmap = plt.get_cmap("cividis", 11)  # Alternatively, use "viridis"
    colors = cmap(np.linspace(0, 1, 11))  # Generate 11 colors

    plt.figure(figsize=(14, 10))

    # Calculate jitter magnitude as 1% of the data range for both x and y
    jitter_x = 0.01 * (df["median_capability"].max() - df["median_capability"].min())
    jitter_y = 0.01 * (df["median_safety"].max() - df["median_safety"].min())

    # Step 3: Compute and Plot Pareto Frontiers and Scatter Points for Each Generation
    for i, (gen, group) in enumerate(generations):
        # Assign a color to the generation
        color = colors[i % len(colors)]

        # Rename columns for clarity
        group = group.rename(columns={"median_capability": "x", "median_safety": "y"})

        # Apply jitter to x and y
        jittered_x = group["x"] + np.random.uniform(
            -jitter_x, jitter_x, size=group.shape[0]
        )
        jittered_y = group["y"] + np.random.uniform(
            -jitter_y, jitter_y, size=group.shape[0]
        )

        # Plot all points in the generation as scatter with jitter
        plt.scatter(
            jittered_x,
            jittered_y,
            color=color,
            alpha=0.7,
            label=f"Generation {gen} - Systems",
            edgecolor="k",
            s=50,
        )

        # Compute Pareto frontier using original (non-jittered) data
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

    # Set axis limits to [0, 1] for both x and y axes
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend(loc="best", fontsize=10, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    population_id = "d44a351c-d454-4d2c-ae74-f7e0e88b9ce8"
    population_id = "dd43526d-9a36-41c3-89bb-2f71c7738040"

    for session in initialize_session():
        population = (
            session.query(Population)
            .order_by(Population.population_timestamp.desc())
            .limit(1)
            .one()
        )
        population_id = population.population_id

        print(population_id)

        # population_id = "32d56f21-dbda-44d9-9d99-980b7eae9898"

        # Suppose you have a list of systems from your DB:
        systems = session.query(System).filter_by(population_id=population_id).all()
        print(len(systems))
        print(systems[0])
        print(population_id)

        plot_pareto_frontiers(systems)
