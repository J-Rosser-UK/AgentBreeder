import sys

sys.path.append("src")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from base import initialize_session, System
import random
from scipy.interpolate import make_interp_spline
from scipy.stats import zscore
from collections import OrderedDict


def plot(systems):
    # Step 1: Extract Relevant Data from Systems
    data = []
    for system in systems:
        if system.system_capability_ci_median == 0:
            continue  # Skip systems with median capability of 0
        data.append(
            {
                "generation_timestamp": system.generation_timestamp,
                "median_capability": system.system_capability_ci_median,
                "ci_lower_capability": system.system_capability_ci_lower,
                "ci_upper_capability": system.system_capability_ci_upper,
                "median_safety": system.system_safety_ci_median,
                "ci_lower_safety": system.system_safety_ci_lower,
                "ci_upper_safety": system.system_safety_ci_upper,
            }
        )

    # Step 2: Convert to DataFrame and Ensure Datetime Format
    df = pd.DataFrame(data)
    df["generation_timestamp"] = pd.to_datetime(df["generation_timestamp"])

    # Step 3: Sort Data by Generation Timestamp
    df = df.sort_values(by="generation_timestamp").reset_index(drop=True)

    # Step 4: Remove Outliers (Optional)
    # Define a function to remove outliers based on Z-score
    def remove_outliers(df, columns, threshold=3):
        df_clean = df.copy()
        for col in columns:
            z_scores = zscore(df_clean[col].dropna())
            mask = np.abs(z_scores) < threshold
            df_clean = df_clean[mask]
        return df_clean

    # Columns to check for outliers
    capability_columns = [
        "median_capability",
        "ci_lower_capability",
        "ci_upper_capability",
    ]
    safety_columns = ["median_safety", "ci_lower_safety", "ci_upper_safety"]

    # Remove outliers from capability and safety data
    df_cap_clean = remove_outliers(df, capability_columns)
    df_safety_clean = remove_outliers(df, safety_columns)

    # Synchronize clean data to ensure both capability and safety are clean
    clean_indices = df_cap_clean.index.intersection(df_safety_clean.index)
    df_clean = df.loc[clean_indices].reset_index(drop=True)

    # Step 5: Aggregate Data by Generation Timestamp
    # Define a function to compute bootstrap confidence intervals
    def bootstrap_median_ci(data, n_bootstrap=1000, ci=95):
        medians = []
        n = len(data)
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            medians.append(np.median(sample))
        lower = np.percentile(medians, (100 - ci) / 2)
        upper = np.percentile(medians, 100 - (100 - ci) / 2)
        return lower, upper

    # Group by generation_timestamp and compute medians and CIs
    summary = (
        df_clean.groupby("generation_timestamp")
        .apply(
            lambda g: pd.Series(
                {
                    "median_capability": np.median(g["median_capability"]),
                    "ci_lower_capability": bootstrap_median_ci(g["median_capability"])[
                        0
                    ],
                    "ci_upper_capability": bootstrap_median_ci(g["median_capability"])[
                        1
                    ],
                    "median_safety": np.median(g["median_safety"]),
                    "ci_lower_safety": bootstrap_median_ci(g["median_safety"])[0],
                    "ci_upper_safety": bootstrap_median_ci(g["median_safety"])[1],
                }
            )
        )
        .reset_index()
    )

    # Step 6: Sort Summary by Generation Timestamp
    summary = summary.sort_values("generation_timestamp").reset_index(drop=True)

    # Step 7: Convert Generation Timestamps to Numeric Values for Spline
    # This is necessary because make_interp_spline requires numeric x-values
    summary["timestamp_numeric"] = summary["generation_timestamp"].map(
        pd.Timestamp.timestamp
    )

    X_numeric = summary["timestamp_numeric"].values
    X_new_numeric = np.linspace(X_numeric.min(), X_numeric.max(), 500)
    X_new = pd.to_datetime(X_new_numeric, unit="s")

    # Step 8: Define Smoothing Function
    def smooth_spline(x_numeric, y, x_new_numeric):
        if len(x_numeric) < 4:
            # Not enough points for cubic spline, fallback to linear interpolation
            y_smooth = np.interp(x_new_numeric, x_numeric, y)
            return y_smooth
        try:
            spline = make_interp_spline(x_numeric, y, k=3)  # Cubic spline
            y_smooth = spline(x_new_numeric)
            return y_smooth
        except Exception as e:
            print(f"Spline fitting failed: {e}. Using linear interpolation instead.")
            y_smooth = np.interp(x_new_numeric, x_numeric, y)
            return y_smooth

    # Step 9: Smooth Median Capability and Confidence Intervals
    median_cap_smooth = smooth_spline(
        X_numeric, summary["median_capability"].values, X_new_numeric
    )
    ci_lower_cap_smooth = smooth_spline(
        X_numeric, summary["ci_lower_capability"].values, X_new_numeric
    )
    ci_upper_cap_smooth = smooth_spline(
        X_numeric, summary["ci_upper_capability"].values, X_new_numeric
    )

    # Step 10: Smooth Median Safety and Confidence Intervals
    median_safety_smooth = smooth_spline(
        X_numeric, summary["median_safety"].values, X_new_numeric
    )
    ci_lower_safety_smooth = smooth_spline(
        X_numeric, summary["ci_lower_safety"].values, X_new_numeric
    )
    ci_upper_safety_smooth = smooth_spline(
        X_numeric, summary["ci_upper_safety"].values, X_new_numeric
    )

    # Step 11: Plotting
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot Median Fitness with Confidence Interval on Left Y-axis
    ax1.set_xlabel("Generation Timestamp")
    ax1.set_ylabel("Median Fitness", color="blue")

    # Plot the smooth median capability line
    (fitness_line,) = ax1.plot(
        X_new,
        median_cap_smooth,
        color="blue",
        label="Median Fitness",
    )

    # Fill the confidence interval
    fitness_ci = ax1.fill_between(
        X_new,
        ci_lower_cap_smooth,
        ci_upper_cap_smooth,
        color="blue",
        alpha=0.2,
        label="95% Confidence Interval (Fitness)",
    )

    # Plot the actual median fitness data points
    fitness_points = ax1.scatter(
        summary["generation_timestamp"],
        summary["median_capability"],
        color="blue",
        alpha=0.6,
        marker="o",
        edgecolor="w",
        label="Median Fitness Points",
    )

    # Fit and plot trend line for capability
    capability_coeffs = np.polyfit(X_numeric, summary["median_capability"], 1)
    capability_trend = np.poly1d(capability_coeffs)
    (fitness_trend_line,) = ax1.plot(
        summary["generation_timestamp"],
        capability_trend(X_numeric),
        color="blue",
        linestyle="--",
        label="Fitness Trend",
    )

    ax1.tick_params(axis="y", labelcolor="blue")

    # Plot Median Safety with Confidence Interval on Right Y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Median Safety", color="red")

    # Plot the smooth median safety line
    (safety_line,) = ax2.plot(
        X_new,
        median_safety_smooth,
        color="red",
        label="Median Safety",
    )

    # Fill the confidence interval
    safety_ci = ax2.fill_between(
        X_new,
        ci_lower_safety_smooth,
        ci_upper_safety_smooth,
        color="red",
        alpha=0.2,
        label="95% Confidence Interval (Safety)",
    )

    # Plot the actual median safety data points
    safety_points = ax2.scatter(
        summary["generation_timestamp"],
        summary["median_safety"],
        color="red",
        alpha=0.6,
        marker="s",
        edgecolor="w",
        label="Median Safety Points",
    )

    # Fit and plot trend line for safety
    safety_coeffs = np.polyfit(X_numeric, summary["median_safety"], 1)
    safety_trend = np.poly1d(safety_coeffs)
    (safety_trend_line,) = ax2.plot(
        summary["generation_timestamp"],
        safety_trend(X_numeric),
        color="red",
        linestyle="--",
        label="Safety Trend",
    )

    ax2.tick_params(axis="y", labelcolor="red")

    # Combine Legends from Both Axes without Duplicates
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine and remove duplicates using OrderedDict
    by_label = OrderedDict(zip(labels1 + labels2, handles1 + handles2))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper left")

    # Enhance X-axis Formatting
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # Add Gridlines for Better Readability
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Set Plot Title
    plt.title(
        "Generational Trends of Fitness and Safety with 95% Bootstrap Confidence Intervals"
    )

    # Adjust Layout for Better Fit
    fig.tight_layout()

    # Display the Plot
    plt.show()


# Example usage
if __name__ == "__main__":
    random.seed(42)
    population_id = "ce611672-5d2b-4577-babe-cf562fab4b1c"

    for session in initialize_session():
        systems = session.query(System).filter_by(population_id=population_id).all()
        plot(systems)
