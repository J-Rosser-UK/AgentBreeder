from inspect_ai.scorer import Score

from inspect_ai.scorer._metric import (
    Metric,
    Score,
    metric,
)

import numpy as np
import logging


@metric
def ci_lower(
    num_bootstrap_samples: int = 1000, confidence_level: float = 0.95
) -> Metric:
    def metric(scores: list[Score]):
        values = np.array([score.as_float() for score in scores])

        n = len(values)

        if n == 0:
            logging.warning("No scores provided. Returning -1 as confidence interval.")
            return -1

        # List to store the means of bootstrap samples
        bootstrap_means = []

        # Generate bootstrap samples and compute the mean for each sample
        for _ in range(num_bootstrap_samples):
            # Resample with replacement
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            # Compute the mean of the bootstrap sample
            bootstrap_mean = np.mean(bootstrap_sample)
            bootstrap_means.append(bootstrap_mean)

        # Convert bootstrap_means to a numpy array for percentile calculation
        bootstrap_means = np.array(bootstrap_means)

        # Compute the lower and upper percentiles for the confidence interval
        lower_percentile = (1.0 - confidence_level) / 2.0
        ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)

        return ci_lower

    return metric


@metric
def ci_upper(
    num_bootstrap_samples: int = 1000, confidence_level: float = 0.95
) -> Metric:
    def metric(scores: list[Score]):
        values = np.array([score.as_float() for score in scores])

        n = len(values)

        if n == 0:
            logging.warning("No scores provided. Returning -1 as confidence interval.")
            return -1

        # List to store the means of bootstrap samples
        bootstrap_means = []

        # Generate bootstrap samples and compute the mean for each sample
        for _ in range(num_bootstrap_samples):
            # Resample with replacement
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            # Compute the mean of the bootstrap sample
            bootstrap_mean = np.mean(bootstrap_sample)
            bootstrap_means.append(bootstrap_mean)

        # Convert bootstrap_means to a numpy array for percentile calculation
        bootstrap_means = np.array(bootstrap_means)

        # Compute the lower and upper percentiles for the confidence interval
        lower_percentile = (1.0 - confidence_level) / 2.0
        upper_percentile = 1.0 - lower_percentile
        ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

        return ci_upper

    return metric


@metric
def median(num_bootstrap_samples: int = 1000, confidence_level: float = 0.95) -> Metric:
    def metric(scores: list[Score]):
        values = np.array([score.as_float() for score in scores])

        n = len(values)

        if n == 0:
            logging.warning("No scores provided. Returning -1 as confidence interval.")
            return -1

        # List to store the means of bootstrap samples
        bootstrap_means = []

        # Generate bootstrap samples and compute the mean for each sample
        for _ in range(num_bootstrap_samples):
            # Resample with replacement
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            # Compute the mean of the bootstrap sample
            bootstrap_mean = np.mean(bootstrap_sample)
            bootstrap_means.append(bootstrap_mean)

        # Convert bootstrap_means to a numpy array for percentile calculation
        bootstrap_means = np.array(bootstrap_means)
        # Compute the median of the bootstrap means
        median = np.median(bootstrap_means)

        return median

    return metric
