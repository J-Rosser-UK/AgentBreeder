import unittest
import numpy as np
import matplotlib.pyplot as plt


def dominates(point_a, point_b):
    """
    Returns True if `point_a` dominates `point_b`, given the rule:
      - We assume we are maximizing both objectives.
      - point_a dominates point_b if:
          point_a[0] >= point_b[0] and point_a[1] >= point_b[1]
        and at least one of these is a strict >.
    """
    return (
        point_a[0] >= point_b[0]
        and point_a[1] >= point_b[1]
        and (point_a[0] > point_b[0] or point_a[1] > point_b[1])
    )


def compute_pareto_front(points):
    """
    Given an array of points (shape N x 2),
    returns the subset of points that form the Pareto front.
    (Naïve O(n^2) approach.)
    """
    pareto_points = []
    n = len(points)
    for i in range(n):
        p_i = points[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            p_j = points[j]
            if dominates(p_j, p_i):
                dominated = True
                break
        if not dominated:
            pareto_points.append(p_i)
    return np.array(pareto_points)


def generate_example_data(num_points=50, seed=0):
    """
    Generates random 2D data to simulate two-objective solutions.
    The data is typically in [0,1] x [0,1].
    """
    np.random.seed(seed)
    data = np.random.rand(num_points, 2)
    return data


class TestParetoFront(unittest.TestCase):

    def test_pareto_front(self):
        # Generate some synthetic 2D data
        data = generate_example_data(num_points=50, seed=42)

        # Compute the Pareto front
        pf = compute_pareto_front(data)

        # Plot the results to visualize
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot all points
        ax.scatter(data[:, 0], data[:, 1], c="gray", alpha=0.6, label="All Points")

        # Plot Pareto front points in a distinct color/marker
        ax.scatter(
            pf[:, 0],
            pf[:, 1],
            c="red",
            marker="o",
            edgecolor="black",
            s=80,
            label="Pareto Front",
        )

        ax.set_title("2D Example Data and Computed Pareto Front")
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.legend(loc="best")

        # Save or show the figure (here we just show it for the sake of example)
        plt.show()

        # We can add some sanity checks in the test
        # For instance, ensure we actually have at least one Pareto point:
        self.assertGreater(len(pf), 0, "Pareto front should not be empty.")

        # Ensure that no point in the Pareto front is dominated by another:
        for i in range(len(pf)):
            for j in range(len(pf)):
                if i == j:
                    continue
                self.assertFalse(
                    dominates(pf[j], pf[i]),
                    "A Pareto front point is dominated, which should not happen.",
                )


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
