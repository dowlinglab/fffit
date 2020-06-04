import numpy as np
import matplotlib.pyplot as plt

from fffit.utils import values_scaled_to_real
from fffit.utils import values_real_to_scaled
from fffit.utils import variances_scaled_to_real


def plot_model_performance(
    models, x_data, y_data, property_bounds, xylim=None
):
    """Plot the predictions vs. result for one or more GP models

    Parameters
    ----------
    models : dict { label : model }
        Each model to be plotted (value, GPFlow model) is provided
        with a label (key, string)
    x_data : np.array
        data to create model predictions for
    y_data : np.ndarray
        correct answer
    property_bounds : array-like
        bounds for scaling density between physical
        and dimensionless values
    xylim : array-like, shape=(2,), optional
        lower and upper x and y limits of the plot
    """

    y_data_physical = values_scaled_to_real(y_data, property_bounds)
    min_xylim = np.min(y_data_physical)
    max_xylim = np.max(y_data_physical)

    for (label, model) in models.items():
        gp_mu, gp_var = model.predict_f(x_data)
        gp_mu_physical = values_scaled_to_real(gp_mu, property_bounds)
        plt.scatter(y_data_physical, gp_mu_physical, label=label, zorder=2.5)
        meansqerr = np.mean(
            (gp_mu_physical - y_data_physical.reshape(-1, 1)) ** 2
        )
        print("Model: {}. Mean squared err: {:.0f}".format(label, meansqerr))
        if np.min(gp_mu_physical) < min_xylim:
            min_xylim = np.min(gp_mu_physical)
        if np.max(gp_mu_physical) > max_xylim:
            max_xylim = np.max(gp_mu_physical)

    if xylim is None:
        xylim = [min_xylim, max_xylim]

    plt.plot(
        np.arange(xylim[0], xylim[1] + 100, 100),
        np.arange(xylim[0], xylim[1] + 100, 100),
        color="xkcd:blue grey",
        label="y=x",
    )

    plt.xlim(xylim[0], xylim[1])
    plt.ylim(xylim[0], xylim[1])
    plt.xlabel("Actual")
    plt.ylabel("Model Prediction")
    plt.legend()
    ax = plt.gca()
    ax.set_aspect("equal", "box")


def plot_slices_temperature(
    models,
    n_params,
    temperature_bounds,
    property_bounds,
    plot_bounds=[220.0, 340.0],
    property_name="property",
):
    """Plot the model predictions as a function of temperature
    Slices are plotted where the values of the other parameters
    are all set to 0.0 --> 1.0 in increments of 0.1
    Parameters
    ----------
    models : dict
        models to plot, key=label, value=gpflow.model
    n_params : int
        number of non-temperature parameters in the model
    temperature_bounds: array-like
        bounds for scaling temperature between physical
        and dimensionless values
    property_bounds: array-like
        bounds for scaling the property between physical
        and dimensionless values
    plot_bounds : array-like, optional
        temperature bounds for the plot
    property_name : str, optional, default="property"
        property name with units for axis label
    """

    n_samples = 100
    vals = np.linspace(plot_bounds[0], plot_bounds[1], n_samples).reshape(
        -1, 1
    )
    vals_scaled = values_real_to_scaled(vals, temperature_bounds)

    for other_vals in np.arange(0, 1.1, 0.1):
        other = np.tile(other_vals, (n_samples, n_params))
        xx = np.hstack((other, vals_scaled))

        for (label, model) in models.items():
            mean_scaled, var_scaled = model.predict_f(xx)
            mean = values_scaled_to_real(mean_scaled, bounds)
            var = variances_scaled_to_real(var_scaled, bounds)

            plt.plot(vals, mean, lw=2, label=label)
            plt.fill_between(
                vals[:, 0],
                mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                alpha=0.3,
            )

        plt.title(f"Other vals = {other_vals:.2f}")
        plt.xlabel("Temperature")
        plt.ylabel(property_name)
        plt.legend()
        plt.show()


def plot_slices_params(
    models,
    parameter,
    parameter_to_idx,
    n_params,
    temperature,
    temperature_bounds,
    property_bounds,
):
    """Plot the model predictions as a function of parameter_idx at the specified temperature
    Parameters
    ----------
    models : dict {"label" : gpflow.model }
        GPFlow models to plot
    parameter : string
        Parameter to vary
    parameter_to_idx : dict { "parameter" : idx }
        dictionary that maps the parameter to the column index
    n_params : int
        number of non-temperature parameters in the model
    temperature : float
        temperature at which to plot the surface
    temperature_bounds: array-like
        bounds for scaling temperature between physical
        and dimensionless values
    property_bounds: array-like
        bounds for scaling property between physical
        and dimensionless values
    """

    parameter_idx = parameter_to_idx[parameter]

    n_samples = 100
    vals = np.linspace(-0.1, 1.1, n_samples).reshape(-1, 1)
    temp_vals = np.tile(temperature, (n_samples, 1))
    temp_vals_scaled = values_real_to_scaled(temp_vals, temperature_bounds)

    for other_vals in np.arange(0, 1.1, 0.1):
        other1 = np.tile(other_vals, (n_samples, parameter_idx))
        other2 = np.tile(other_vals, (n_samples, n_params - 1 - parameter_idx))
        xx = np.hstack((other1, vals, other2, temp_vals_scaled))

        for (label, model) in models.items():
            mean_scaled, var_scaled = model.predict_f(xx)
            mean = values_scaled_to_real(mean_scaled, density_bounds)
            var = variances_scaled_to_real(var_scaled, density_bounds)

            plt.plot(vals, mean, lw=2, label=label)
            plt.fill_between(
                vals[:, 0],
                mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                alpha=0.3,
            )

        math_parameter = "$\\" + parameter + "$"
        plt.title(
            f"{math_parameter} at T = {temperature:.0f} K. Other vals = {other_vals:.2f}."
        )
        plt.xlabel("$\\" + parameter + "$")
        plt.ylabel(property_name)
        plt.legend()
        plt.show()


def plot_model_vs_test(
    models,
    param_values,
    train_points,
    test_points,
    temperature_bounds,
    property_bounds,
    plot_bounds=[220.0, 340.0],
    property_name="property",
):
    """Plots the GP model(s) as a function of temperature with all other parameters
    taken as param_values. Overlays training and testing points with the same
    param_values.

    Parameters
    ----------
    models : dict {"label" : gpflow.model }
        GPFlow models to plot
    param_values : np.ndarray, shape=(n_params)
        The parameters at which to evaluate the GP model
    train_points : np.ndarray, shape=(n_points, 2)
        The temperature (scaled) and property (scaled) of each training point
    test_points : np.ndarray, shape=(n_points, 2)
        The temperature (scaled) and property (scaled) of each test point
    temperature_bounds: array-like
        bounds for scaling temperature between physical
        and dimensionless values
    property_bounds: array-like
        bounds for scaling property between physical
        and dimensionless values
    plot_bounds : array-like, optional
        temperature bounds for the plot
    property_name : str, optional, default="property"
        property name with units for axis label
    """

    n_samples = 100
    vals = np.linspace(plot_bounds[0], plot_bounds[1], n_samples).reshape(
        -1, 1
    )
    vals_scaled = values_real_to_scaled(vals, temperature_bounds)

    other = np.tile(param_values, (n_samples, 1))
    xx = np.hstack((other, vals_scaled))

    for (label, model) in models.items():
        mean_scaled, var_scaled = model.predict_f(xx)

        mean = values_scaled_to_real(mean_scaled, property_bounds)
        var = variances_scaled_to_real(var_scaled, property_bounds)
        plt.plot(vals, mean, lw=2, label="GP model" + label)
        plt.fill_between(
            vals[:, 0],
            mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
            mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
            alpha=0.25,
        )

    if train_points.shape[0] > 0:
        md_train_temp = values_scaled_to_real(
            train_points[:, 0], temperature_bounds
        )
        md_train_property = values_scaled_to_real(
            train_points[:, 1], property_bounds
        )
        plt.plot(
            md_train_temp, md_train_property, "s", color="black", label="Train"
        )
    if test_points.shape[0] > 0:
        md_test_temp = values_scaled_to_real(
            test_points[:, 0], temperature_bounds
        )
        md_test_property = values_scaled_to_real(
            test_points[:, 1], property_bounds
        )
        plt.plot(md_test_temp, md_test_property, "ro", label="Test")

    plt.xlabel("Temperature")
    plt.ylabel(property_name)
    plt.legend()
    plt.show()
