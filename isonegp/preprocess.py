import numpy as np
from .plot import plot_crosscorr, plot_autocorr


def pretrain_checks(
        training: np.ndarray, testing: np.ndarray, run_folder: str, savefile_prefix: str = ""
) -> None:
    """
    Performs a growing list of processing functions on the training and testing
     data provided and plots their results
    
    Parameters
    ----------
    training: np.ndarray
    Training data as a 1D numpy array

    testing: np.ndarray
    Testing data as a 1D numpy array

    run_folder: str
    Folder where processing functions will save figures generated

    Returns
    -------
    None
    """
    plot_args = {
        'is_save': True,
        'run_prefix': run_folder,
        'save_filename': f'{savefile_prefix}train_test_xcorr',
    }
    plot_crosscorr(
        training,
        testing,
        'Crosscorrelation between training data and testing data',
        **plot_args
    )

    plot_args['save_filename'] = f'{savefile_prefix}training_data_autocorreltation'
    plot_autocorr(training, 'Autocorrelation with training data', **plot_args)

    plot_args['save_filename'] = f'{savefile_prefix}testing_data_autocorreltation'
    plot_autocorr(testing, 'Autocorrelation with testing data', **plot_args)


def normalize(raw_data: np.ndarray) -> np.ndarray:
    normalized_data = raw_data.copy()
    mu = np.mean(normalized_data)
    sigma = np.std(normalized_data, mean=mu)
    for i in range(normalized_data.size):
        normalized_data[i] = (normalized_data[i] - mu) / sigma
    return normalized_data
