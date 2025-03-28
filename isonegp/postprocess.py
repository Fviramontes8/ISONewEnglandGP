import numpy as np


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Mean Absolute Percentage Error, assumes arguments are same size

    Parameters
    ----------
    y_true: np.ndarray
    Regressors from a validation/testing dataset

    y_pred: np.ndarray
    Predicted regressors from a Gaussian Process model

    Returns
    -------
    A float representing a percentage error, lower is better
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
