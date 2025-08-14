import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


def generate_observations(
    N=100, dimensions=1, K=2, mean_bounds=(0, 10), std_bounds=(0.5, 2)
):
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

    return source_kwargs, weights, np.array(true_labels), np.array(observations)


def decode_distribution(distribution):
    """
    Helper function to unpack the two types of distributions:

    1) kwargs for np.random.normal:
        {"loc": mean: "scale": stddev}
    2) array
        [mean, [mean], stddev, [stddev]]
    """
    if isinstance(distribution, dict):
        return (distribution["loc"], distribution["scale"])
    else:
        dim = len(distribution) // 2
        return distribution[:dim], distribution[dim:]


def visualize_observations(
    axis,
    observations=None,
    labels=None,
    distributions=None,
    marker="o",
    cmap_name="tab10",
):

    if distributions is not None:
        if cmap_name == "grey":
            # Special case to allow just plotting a constant color
            cmap = lambda x: (0.4, 0.4, 0.4, 1)
        else:
            cmap = plt.get_cmap(cmap_name, len(distributions))
        for i, distribution in enumerate(distributions):
            mean, stddev = decode_distribution(distribution)
            if len(mean) == 1:
                x = np.linspace(mean - 3 * stddev, mean + 3 * stddev)
                y = multivariate_normal.pdf(x, mean=mean, cov=stddev**2)
                axis.plot(x, y, color=cmap(i))
            else:
                for scale, opacity in ([1, 0.35], [2, 0.2]):
                    axis.add_patch(
                        # Note that this is total width, 2x the desired stddev "radius"
                        Ellipse(
                            xy=mean,
                            width=2 * scale * stddev[0],
                            height=2 * scale * stddev[1],
                            color=cmap(i)[:3] + (opacity,),
                        )
                    )

    if observations is not None:

        if observations.shape[1] == 1:
            observations = np.hstack([observations, np.zeros_like(observations)])
        elif len(observations.shape) != 2 or observations.shape[1] != 2:
            raise NotImplementedError(observations.shape)

        if labels is None:
            colors = ["k"] * len(observations)
        else:
            unique = np.unique(labels)
            cmap = plt.get_cmap(cmap_name, len(unique))
            colors = [cmap(i) for i in labels]

        axis.scatter(
            observations[:, 0], observations[:, 1], color=colors, marker=marker
        )


def visualize_weights(axis, observation, weights, distributions, cmap_name="tab10"):

    # Handle 1D vs 2D
    if len(observation) == 1:
        ox = observation
        oy = np.random.random() * -0.2
    else:
        ox, oy = observation

    cmap = plt.get_cmap(cmap_name, len(distributions))
    for j, (weight, distribution) in enumerate(zip(weights, distributions)):

        # Handle 1D vs 2D
        if len(observation) == 1:
            dx, _ = decode_distribution(distribution)
            dy = oy
        else:
            (dx, dy), _ = decode_distribution(distribution)

        axis.plot([ox, dx], [oy, dy], color=cmap(j), linewidth=max(10 * weight, 0.2))

    axis.plot(ox, oy, "ko")


def visualize_assignments(axis, observations, weights, cmap_name="tab10", marker="o"):

    if observations.shape[1] == 1:
        observations = np.hstack([observations, np.zeros_like(observations)])
    elif len(observations.shape) != 2 or observations.shape[1] != 2:
        raise NotImplementedError(observations.shape)

    cmap = plt.get_cmap(cmap_name, weights.shape[1])
    colors = []
    for o_weight in weights:
        colors.append(cmap(np.argmax(o_weight)))

    axis.scatter(observations[:, 0], observations[:, 1], color=colors, marker=marker)


def likelihood(sample, mean, stddev):

    # Turn [cov x, cov y] vector into [[cov x, 0], [0, cov y]]
    if stddev.ndim == 1:
        stddev = np.diag(stddev)

    return multivariate_normal.pdf(sample, mean=mean, cov=stddev**2)


def e_step(observations, K, mu, sigma, psi):

    # i x j, aka len(observations) x len(sources)
    e_weights = np.zeros((len(observations), K))

    # First build the numerators
    for i in range(len(observations)):
        for j in range(K):
            e_weights[i, j] = likelihood(observations[i], mu[j], sigma[j]) * psi[j]

    # Then normalize by the denominator
    for i in range(len(observations)):
        e_weights[i, :] /= np.sum(e_weights[i, :])

    return e_weights


def m_step(observations, weights, mu_old, sigma_old, psi_old, epsilon=1e-6):

    # Average the weights over the observations
    psi_new = np.sum(weights, axis=0) / len(weights)

    # For each source, minimize the negative log likelihood of the related parameters
    mu_new = []
    sigma_new = []
    costs = []
    dim = len(mu_old[0])
    for j in range(weights.shape[1]):

        # Annoyingly, I think this needs to be defined in situ so it can take only
        # the argument it needs to minimize while still receiving other variables
        def neg_log_likelihood(x):
            sumval = 0
            for i, weight in enumerate(weights[:, j]):
                if weight > epsilon:
                    value = likelihood(observations[i], mean=x[:dim], stddev=x[dim:])
                    # Make sure the value has a minimum to avoid invalid log calls
                    sumval += weight * np.log(np.clip(value, 1e-200, None))
            return -sumval

        result = minimize(
            neg_log_likelihood, np.hstack([mu_old[j], sigma_old[j]]), method="BFGS"
        )
        mu_new.append(result.x[:dim])
        sigma_new.append(result.x[dim:])
        costs.append(result.fun)

    return np.array(mu_new), np.array(sigma_new), np.array(psi_new), np.sum(costs)
