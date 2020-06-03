import numpy as np


def values_real_to_scaled(values, bounds):
    """Convert values in physical units to values scaled by bounds

    Parameters
    ----------
    values : array_like, shape=(n,m)
        Input values (unscaled)
    bounds : array_like, shape=(m,2)
        Bounds to scale `values`. Lower bound is 0 and upper bound
        is 1 in `scaled_values`.

    Returns
    -------
    scaled_values : np.ndarray, shape=(n,m)
        The values scaled by `bounds`

    Notes
    -----
    The `bounds` define the 0 and 1 limits of the `scaled_values`.
    The `values` may exceed the bounds; in this case the
    `scaled_values` will have values < 0 or > 1.
    """
    values, bounds = _clean_bounds_values(values, bounds)
    return (values - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def values_scaled_to_real(scaled_values, bounds):
    """Convert scaled values to values in physical units

    Parameters
    ----------
    scaled_values : array_like, shape=(n,m)
        Input values (scaled)
    bounds : array_like, shape=(m,2)
        Bounds to scale `values`. Lower bound is 0 and upper bound
        is 1 in `scaled_values`.

    Returns
    -------
    values : np.ndarray, shape=(n,m)
        The values in unscaled units

    Notes
    -----
    The `bounds` define the 0 and 1 limits of the `scaled_values`.
    The `scaled_values` may exceed the 0 and 1; in this case the
    `values` will have values < lower bound or > upper bound.
    """
    scaled_values, bounds = _clean_bounds_values(scaled_values, bounds)
    return scaled_values * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def variances_scaled_to_real(scaled_variances, bounds):
    """Convert variance in scaled dimensionless values to physical units

    Parameters
    ----------
    scaled_variances : array_like, shape=(n,m)
        Input variances (scaled)
    bounds : array_like, shape=(m,2)
        Bounds to scale `scaled_variances`. Lower bound is 0 and upper bound
        is 1 in `scaled_values`.

    Returns
    -------
    real_vars : np.ndarray, shape=(n,m)
        The variance values in unscaled units
    """
    scaled_variances, bounds = _clean_bounds_values(scaled_variances, bounds)

    if (scaled_variances < 0.0).any():
        raise ValueError("Variance cannot be less than zero")

    return scaled_variances * (bounds[:, 1] - bounds[:, 0]) ** 2


def _clean_bounds_values(values, bounds):
    values = np.asarray(values)
    bounds = np.asarray(bounds)
    bounds = bounds.reshape(-1, 2)

    if not (bounds[:, 0] < bounds[:, 1]).all():
        raise ValueError(
            "Lower bound must always be less than the upper bound."
        )

    if bounds.shape[0] == 1:
        values = values.reshape(-1, 1)
    else:
        if len(values.shape) != 2 or values.shape[1] != bounds.shape[0]:
            raise ValueError(
                "Shapes of `values` and `bounds` must be consistent"
            )

    return values, bounds
