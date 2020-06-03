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


def shuffle_and_split(df, n_params, property_name, fraction_train=0.8, shuffle_seed=None):
    """Randomly shuffle the DataFrame and extracts the train and test sets

    The first n_params columns of the dataframe must be the parameters,
    the n_params+1 column must be the temperature. The other quantities
    can be in any column.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe with the
    n_params : int
        Number of parameters (not including temperature) in the model
    property_name : string
        Name of the property to extract from the pandas dataframe
    fraction_train : float
        Fraction to use as training data. The remainder will be used for testing
    shuffle_seed : int, optional
        seed for random number generator for shuffle

    Returns
    -------
    x_train : np.ndarray
        Training inputs
    y_train : np.ndarray
        Training results
    x_test : np.ndarray
        Testing inputs
    y_test : np.ndarray
        Testing results
    """
    if fraction_train <= 0.0 or fraction_train >= 1.0:
        raise ValueError("`fraction_train` must be between 0 and 1.")
    else:
        fraction_test = 1.0 - fraction_train

    try:
        prp_idx = df.columns.get_loc(property_name)
    except KeyError:
        raise ValueError(
            "`property_name` does not match any headers of `df`"
        )
    data = df.values
    total_entries = data.shape[0]
    train_entries = int(total_entries * fraction_train)
    # Shuffle the data before splitting train/test sets
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    x_train = data[:train_entries, :n_params+1].astype(np.float64)
    y_train = data[:train_entries, prp_idx].astype(np.float64)
    x_test = data[train_entries:, : n_params + 1].astype(np.float64)
    y_test = data[train_entries:, prp_idx].astype(np.float64)

    return x_train, y_train, x_test, y_test
