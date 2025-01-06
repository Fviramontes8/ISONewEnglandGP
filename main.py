from gpytorch.likelihoods import GaussianLikelihood
import numpy as np
from torch import linspace, Tensor
from torch.optim import Adam

from isonegp.gpmodel import GaussianProcess, CustomGPModel
from isonegp.preprocess import pretrain_checks, normalize
from isonegp.plot import (
    plot_cat_data,
    plot_data,
    plot_overlapping_data,
    plot_gp_inference,
)
from isonegp.session_generator import create_run_folder


def main():
    # The year can change from 2011 to 2016
    new_england_load_demand_data = np.load("data/ISONE_CA_DEMAND_2011.npy")
    print(new_england_load_demand_data.shape)
    # 16 days
    training_indices = 24 * 16
    # 4 days
    testing_indices = 24 * 4
    total_indices = training_indices + testing_indices

    demand_training_data = np.array(
        new_england_load_demand_data[:training_indices],
    )
    demand_testing_data = np.array(
        new_england_load_demand_data[training_indices:total_indices]
    )
    demand_description = "Non-PTF Load Demand"
    demand_units = "MegaWatts"

    current_run_folder = create_run_folder()
    b_save_figures = True
    plot_cat_data(
        [demand_training_data, demand_testing_data],
        ["c", "m"],
        ["Training data", "Testing data"],
        demand_description,
        "Hours",
        demand_units,
        b_save_figures,
        current_run_folder,
        "training_testing_data_fullview",
    )
    pretrain_checks(
        demand_training_data, demand_testing_data, current_run_folder
    )

    normalized_training_data = normalize(demand_training_data)
    normalized_testing_data = normalize(demand_testing_data)
    plot_cat_data(
        [normalized_training_data, normalized_testing_data],
        ["c", "m"],
        ["Training data", "Testing data"],
        demand_description,
        "Hours",
        demand_units,
        b_save_figures,
        current_run_folder,
        "normalized_training_testing_data_fullview",
    )

    normalized_filename_prefix = "normalized_"
    pretrain_checks(
        normalized_training_data,
        normalized_testing_data,
        current_run_folder,
        normalized_filename_prefix,
    )

    gpmodel_params = {
        "train_x": linspace(0, training_indices - 1, training_indices),
        "train_y": Tensor(normalized_training_data),
        "likelihood": GaussianLikelihood(),
    }
    gpmodel = CustomGPModel(**gpmodel_params)
    gpopt = Adam(gpmodel.parameters(), lr=0.1)
    gp_params = {
        "gpmodel": gpmodel,
        "likelihood": gpmodel_params["likelihood"],
        "optimizer": gpopt,
        "train_x": gpmodel_params["train_x"],
        "train_y": gpmodel_params["train_y"],
    }
    model = GaussianProcess(**gp_params)
    loss, noise = model.train(70, True)

    plot_data(
        loss,
        "Loss during training",
        "Training epochs",
        "Training loss",
        b_save_figures,
        current_run_folder,
        "gp_training_loss",
    )

    gp_pred = model(Tensor(normalized_testing_data))
    plot_overlapping_data(
        [gp_pred.mean.numpy(), normalized_testing_data],
        "Trained model prediction v. testing data",
        "Time (Hours)",
        "Normalized MegaWatts",
        b_save_figures,
        current_run_folder,
        "gp_pred_v_testing_data",
    )

    lower, upper = gp_pred.confidence_region()
    plot_gp_inference(
        normalized_training_data,
        normalized_testing_data,
        gp_pred.mean.numpy(),
        lower.numpy(),
        upper.numpy(),
        b_save_figures,
        current_run_folder,
        "gp_inference",
    )


if __name__ == "__main__":
    main()
