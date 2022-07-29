import numpy as np
import math
import matplotlib.pyplot as plt

import torch
import gpytorch

import PlotUtils as pu
import GPyTorchUtils as gptu
import SignalProcessor as sp
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

def one_hour_prediction(input_data):
    train_x, train_y, test_x, test_y = gptu.torch_one_hour_data_split(input_data)
    test_x = test_x.view(1, -1)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = gptu.LinearGPModel(train_x, train_y, likelihood)
    optimizer = torch.optim.Adam(
        [
            {"params" : gp_model.parameters()},
        ],
        lr = 0.1
    )

    train_loss = gptu.TorchTrain(
        train_x,
        train_y,
        gp_model,
        likelihood,
        optimizer,
        50,
        True
    )
    pu.general_plot(train_loss, "Loss over time")
    pred = gptu.TorchTest(test_x, gp_model, likelihood)
    print(f"{test_y.item()}")
    print(f"{pred.mean.numpy()}")

def one_day_prediction(input_data):
    train_x, test_x = gptu.torch_one_hour_data_split(
        input_data,
        0,
        False
    )
    train_y, test_y = gptu.torch_one_hour_target_split(input_data)
    for i in range(1, 24):
        temp_train_y, temp_test_y = gptu.torch_one_hour_target_split(
            input_data,
            i
        )
        train_y = torch.vstack([train_y, temp_train_y])
        test_y = torch.vstack([test_y, temp_test_y])

    test_y = torch.squeeze(test_y)
    test_x = torch.unsqueeze(test_x, 0)


    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = gptu.OneDayLinearGP(train_x, train_y, likelihood)

    optimizers = [
        torch.optim.Adam(
            [
                {"params" : i.parameters(),}
            ],
            lr = 0.1
        )
        for i in gp_model.models
    ]
    gp_model.train(optimizers, 30)
    gp_model.test(test_x)

    local_pred = [i.mean.numpy() for i in gp_model.pred]

    print(f"\nMAPE: {sp.mape_test(np.array(test_y), np.array(local_pred)):.3f}")
    print(f"Mean absolute error: {mae(test_y, local_pred):.3f}")
    print(f"r^2 value: {r2_score(test_y, local_pred):.3f}")

    confi_sigma1 = gptu.confirm_confidence_region(
        local_pred,
        test_y,
        gp_model.upper_sigma,
        gp_model.lower_sigma
    ) * 100
    print(f"{confi_sigma1:.1f}% is contained in a confidence interval", end=' ')
    print(f"of 1 standard deviation")

    confi_sigma2 = gptu.confirm_confidence_region(
        local_pred,
        test_y,
        gp_model.upper_two_sigma,
        gp_model.lower_two_sigma
    ) * 100
    print(f"{confi_sigma2:.1f}% is contained in a confidence interval", end=' ')
    print(f"of 2 standard deviations")

    pred_title = "Linear Gaussian process prediction one day ahead"
    xlabel = "Time (hours)"
    ylabel = "Load demand (MW)"

    gp_model.plot_pred(test_y, pred_title, xlabel, ylabel)

def multitask_one_day_prediction(input_data):
    pass

def main():
    # The year can change from 2011 to 2015
    new_england_load_demand_data = np.load("data/ISONE_CA_DEMAND_2011.npy")
    data_cutoff = (24 * 12) #192
    feature_data = np.array(
        new_england_load_demand_data[:data_cutoff],
    )
    feature_description = "Non-PTF Load Demand"
    feature_units = "MW"

    print(f"Data length: {len(feature_data)}")
    one_hour_prediction(feature_data)
    one_day_prediction(feature_data)

    train_x, test_x = gptu.torch_one_hour_data_split(
        feature_data,
        0,
        False
    )
    train_y, test_y = gptu.torch_one_hour_target_split(feature_data)
    for i in range(1, 24):
        temp_train_y, temp_test_y = gptu.torch_one_hour_target_split(
            feature_data,
            i
        )
        train_y = torch.vstack([train_y, temp_train_y])
        test_y = torch.vstack([test_y, temp_test_y])

    # train_y = torch.transpose(train_y, 0, 1)
    # train_x = torch.transpose(train_x, 0, 1)
    test_y = torch.squeeze(test_y)
    # test_x = torch.unsqueeze(test_x, 0)

    print(f"Train x shape: {train_x.shape}")
    print(f"Train y shape: {train_y.shape}")
    print(f"Test x shape: {test_x.shape}")
    print(f"Test y shape: {test_y.shape}")

    train_rank = torch.linalg.matrix_rank(train_x)
    training_iter = 50
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=train_y.shape[0]
    )

    print(f"Number of tasks: {train_y.shape[0]}")
    print(f"Matrix rank of train x: {train_rank}")

    model = gptu.MultitaskGPModel(
        train_x,
        train_y,
        likelihood,
        train_rank,
        train_y.shape[0]
    )
    opt = torch.optim.Adam(
        model.parameters(),
        lr=0.1
    )

    loss = gptu.TorchTrain(
        train_x,
        train_y,
        model,
        likelihood,
        opt,
        training_iter,
        True
    )
    pu.general_plot(loss, "Training loss", "Time", "Loss")

if __name__ == "__main__":
    main()
