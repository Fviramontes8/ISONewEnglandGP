import numpy as np

from isonegp.preprocess import pretrain_checks
from isonegp.plot import plot_cat_data
from isonegp.session_generator import create_run_folder


def main():
    # The year can change from 2011 to 2016
    new_england_load_demand_data = np.load('data/ISONE_CA_DEMAND_2011.npy')
    # 12 days
    training_indices = 24 * 12
    # 4 days
    testing_indices = 24 * 4
    total_indices = training_indices + testing_indices

    demand_training_data = np.array(
        new_england_load_demand_data[:training_indices],
    )
    demand_testing_data = np.array(
        new_england_load_demand_data[training_indices:total_indices]
    )
    demand_description = 'Non-PTF Load Demand'
    demand_units = 'MegaWatts'

    current_run_folder = create_run_folder()
    plot_cat_data(
        [demand_training_data, demand_testing_data],
        ['c', 'm'],
        ['Training data', 'Testing data'],
        demand_description,
        'Hours',
        demand_units,
        True,
        current_run_folder,
        'training_testing_data_fullview',
    )
    pretrain_checks(
        demand_training_data, demand_testing_data, current_run_folder
    )


if __name__ == '__main__':
    main()
