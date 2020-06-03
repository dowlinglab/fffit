def params_real_to_scaled(params, bounds):
    """Convert sample with physical units to values between 0 and 1"""
    return (params - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def params_scaled_to_real(params, bounds):
    """Convert sample with values between 0 and 1 to physical units"""
    return params * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def values_real_to_scaled(values, bounds):
    """Convert temperature or density with physical units to value between 0 and 1"""
    return (values - bounds[0]) / (bounds[1] - bounds[0])


def values_scaled_to_real(values, bounds):
    """Convert temperature or density with values between 0 and 1 to physical units"""
    return values * (bounds[1] - bounds[0]) + bounds[0]


def means_scaled_to_real(means, bounds):
    """Convert temperature or density mean scaled dimensionless values to physical units"""
    return means * (bounds[1] - bounds[0]) + bounds[0]


def vars_scaled_to_real(ivars, bounds):
    """Convert temperature or density variance scaled dimensionless values to physical units"""
    return ivars * (bounds[1] - bounds[0]) ** 2


def values_real_to_scaled(values, bounds):
    """Convert values in physical units to values scaled by bounds

    Parameters
    ----------
    values : array_like, shape=(n,m)
        Input values (unscaled)
    bounds : array_like, shape=(m,2)
        Bounds to scale `values` by. Lower bound is 0 and upper bound
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
    values = np.asarray(values)
    bounds = np.asarray(bounds)

    bounds.reshape(-1,2)
    values.reshape(-1, bounds.shape[0])

    if values.shape[1] != bounds.shape[0]:
        raise ValueError(
            "Shapes of `values` and `bounds` must be consistent"
        )

    return (values - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def values_scaled_to_real(values, bounds):
    """Convert temperature or density with values between 0 and 1 to physical units"""
    return values * (bounds[1] - bounds[0]) + bounds[0]

