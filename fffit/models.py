import gpflow
import numpy as np

from gpflow.utilities import print_summary


def run_gpflow_scipy(x_train, y_train, kernel, fmt="notebook"):
    """Create and train a GPFlow model

    Parameters
    ----------
    x_train : np.ndarray, shape=(n_samples, n_parameters)
        The x training data
    y_train : np.ndarray, shape=(n_samples, 1)
        The y training data
    kernel : string
        Kernel to use for the GP model
    fmt : string, optional, default="notebook"
        The formatting type for the GPFlow print_summary
    """
    # Create the model
    model = gpflow.models.GPR(
        data=(x_train, y_train.reshape(-1, 1)),
        kernel=kernel,
        mean_function=gpflow.mean_functions.Linear(
            A=np.zeros(x_train.shape[1]).reshape(-1, 1)
        ),
    )

    # Print initial values
    print_summary(model, fmt=fmt)

    # Optimize model with scipy
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(model.training_loss, model.trainable_variables)

    # Print the optimized values
    print_summary(model, fmt="notebook")

    # Return the model
    return model
