import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def generate_observations(N=100, dimensions=1, K=2, mean_bounds=(0, 10), std_bounds=(0.5, 2)):
    """
    Generate N samples of data for a mixture of K Gaussians of the specified dimension.
    """

    # Generate K normal distributions with different locations (means) and scales
    # (standard deviations)
    source_kwargs = [
        {
            "loc": np.random.uniform(*mean_bounds, size=dimensions),
            "scale": np.random.uniform(*std_bounds, size=dimensions),
        }
        for _ in range(K)
    ]

    # Generate weights for the different distributions (must sum to 1)
    weights = np.array([np.random.uniform(0.25, 1) for _ in range(K)])
    weights /= np.sum(weights)

    # Draw from the weighted normal distributions N times
    true_labels = []
    observations = []
    for _ in range(N):
        i = np.random.choice(range(K), p=weights)
        true_labels.append(i)
        observations.append(np.random.normal(**source_kwargs[i]))

    return source_kwargs, weights, true_labels, np.array(observations)


def visualize_observations(axis, observations, labels=None, marker="o"):

    if len(observations.shape) == 1:
        observations = np.vstack([observations, np.zeros_like(observations)]).T
    elif observations.shape[1] == 1:
        observations = np.hstack([observations, np.zeros_like(observations)])
    elif len(observations.shape) > 2 or observations.shape[1] != 2:
        raise NotImplementedError(observations.shape)

    if labels is None:
        colors = ["k"] * len(observations)
    else:
        unique = np.unique(labels)
        cmap = plt.get_cmap("tab10", len(unique))
        colors = [cmap(i) for i in labels]

    axis.scatter(observations[:, 0], observations[:, 1], color=colors, marker=marker)


def likelihood(sample, mean, covariance):

    # Turn [cov x, cov y] vector into [[cov x, 0], [0, cov y]]
    if covariance.ndim == 1:
        covariance = np.diag(covariance)

    return multivariate_normal.pdf(sample, mean=mean, cov=covariance)
