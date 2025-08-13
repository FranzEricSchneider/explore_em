import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import i0  # Bessel function for Von Mises error


def rotate(vector, angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    R = np.array([[cos, -sin], (sin, cos)])
    rotated = R @ vector.reshape(-1, 1)
    return rotated.squeeze()


def generate_observations(
    n_bodies=10,
    n_observers=20,
    xy_bounds=(0, 10),
    std=np.deg2rad(10),
    sight_bounds=(1, 3),
):
    """
    Generate N samples of data for a number of observers viewing a number of
    bodies.

    TODO: Add in the idea of sight limits
    TODO: Add in the idea of false positives and false negatives
    """

    bodies = np.random.uniform(*xy_bounds, size=(n_bodies, 2))
    observers = np.random.uniform(*xy_bounds, size=(n_observers, 2))

    # Loop over all combinations of bodies and observers and make sightings
    observer_ids = []
    observed_angles = []
    labels = []

    for i, observer in enumerate(observers):
        for j, body in enumerate(bodies):
            observer_ids.append(i)
            labels.append(j)

            # Measure the angle as a unit vector, keep if it is within sight range
            vector = body - observer
            length = np.linalg.norm(vector)
            if length < sight_bounds[0] or length > sight_bounds[1]:
                continue
            unit_vector = vector / length

            # Then add measurement noise
            observed_angles.append(
                rotate(unit_vector, np.random.normal(loc=0, scale=std))
            )

    return (
        bodies,
        observers,
        np.array(observer_ids),
        np.array(labels),
        np.array(observed_angles),
    )


def visualize_observations(
    axis,
    observers=None,
    observer_ids=None,
    observations=None,
    labels=None,
    bodies=None,
    cmap_name="tab10",
    bodysize=40,
):

    if bodies is not None:
        if cmap_name == "grey":
            # Special case to allow just plotting a constant color
            cmap = lambda x: (0.4, 0.4, 0.4, 1)
        else:
            cmap = plt.get_cmap(cmap_name, len(bodies))
        colors = [cmap(i) for i in range(len(bodies))]

        axis.scatter(bodies[:, 0], bodies[:, 1], marker="^", color=colors, s=bodysize)

    if observers is not None and observer_ids is not None and observations is not None:

        if labels is None:
            colors = "black"
        else:
            unique = np.unique(labels)
            cmap = plt.get_cmap(cmap_name, len(unique))
            colors = [cmap(i) for i in labels]

        axis.quiver(
            observers[observer_ids][:, 0],
            observers[observer_ids][:, 1],
            observations[:, 0],  # dx
            observations[:, 1],  # dy
            color=colors,
            angles="xy",
            scale_units="xy",
            scale=1,
        )

    if observers is not None:
        axis.scatter(observers[:, 0], observers[:, 1], color="k", s=5)


def visualize_weights(axis, observer, observation, weights, bodies, cmap_name="tab10"):

    axis.quiver(
        observer[0],
        observer[1],
        observation[0],  # dx
        observation[1],  # dy
        angles="xy",
        scale_units="xy",
        scale=1,
    )

    cmap = plt.get_cmap(cmap_name, len(bodies))
    for j, (weight, body) in enumerate(zip(weights, bodies)):
        ox, oy = observer + observation
        bx, by = body
        axis.plot([ox, bx], [oy, by], color=cmap(j), linewidth=max(10 * weight, 0.2))

    axis.plot(observer[0], observer[1], "ko")


def visualize_assignments(
    axis, weights, observers, observer_ids, observations, cmap_name="tab10"
):

    if observations.shape[1] == 1:
        observations = np.hstack([observations, np.zeros_like(observations)])
    elif len(observations.shape) != 2 or observations.shape[1] != 2:
        raise NotImplementedError(observations.shape)

    cmap = plt.get_cmap(cmap_name, weights.shape[1])
    colors = []
    for o_weight in weights:
        colors.append(cmap(np.argmax(o_weight)))

    axis.quiver(
        observers[observer_ids][:, 0],
        observers[observer_ids][:, 1],
        observations[:, 0],  # dx
        observations[:, 1],  # dy
        color=colors,
        angles="xy",
        scale_units="xy",
        scale=1,
    )

    axis.scatter(observers[:, 0], observers[:, 1], color="k")


def von_mises_C(kappa):
    """
    TODO
    """
    return 1.0 / (2 * np.pi * i0(kappa))


def ray_likelihood(observer, ray, body, kappa):
    """Von Mises probability for direction from observer to body."""
    vector = body - observer
    unit_vector = vector / np.linalg.norm(vector)
    return von_mises_C(kappa) * np.exp(kappa * np.dot(unit_vector, ray))


def e_step(K, mu, psi, kappa, observers, observer_ids, observations):

    # i x j, aka len(observations) x len(sources)
    e_weights = np.zeros((len(observations), K))

    # First build the numerators
    for i in range(len(observations)):
        for j in range(K):
            e_weights[i, j] = psi[j] * ray_likelihood(
                observers[observer_ids[i]],
                observations[i],
                mu[j],
                kappa,
            )

    # Then normalize by the denominator
    for i in range(len(observations)):
        e_weights[i, :] /= np.sum(e_weights[i, :])

    return e_weights


def m_step(
    weights, mu_old, psi_old, kappa, observers, observer_ids, observations, epsilon=1e-6
):

    # Average the weights over the observations
    psi_new = np.sum(weights, axis=0) / len(weights)

    # For each source, minimize the negative log likelihood of the related parameters
    mu_new = []
    for j in range(weights.shape[1]):

        # Annoyingly, I think this needs to be defined in situ so it can take only
        # the argument it needs to minimize while still receiving other variables
        def neg_log_likelihood(x):
            sumval = 0
            for i, weight in enumerate(weights[:, j]):
                if weight > epsilon:
                    value = ray_likelihood(
                        observer=observers[observer_ids[i]],
                        ray=observations[i],
                        body=x,
                        kappa=kappa,
                    )
                    # Make sure the value has a minimum to avoid invalid log calls
                    sumval += weight * np.log(np.clip(value, 1e-200, None))
            return -sumval

        result = minimize(neg_log_likelihood, mu_old[j], method="BFGS")
        mu_new.append(result.x)

    return np.array(mu_new), np.array(psi_new)
