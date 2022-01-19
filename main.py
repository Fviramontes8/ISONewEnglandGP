import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

import torch
import gpytorch

import PlotUtils as pu
import GPyTorchUtils as gptu

def read_xls(xls_filename, sheet):
    return pd.read_excel(xls_filename, sheet_name=sheet)

def plot_ISO_features(feature_data, feature_descriptions, feature_units):
    for data, description, units in zip(
        feature_data, 
        feature_descriptions, 
        feature_units
    ): 
        pu.general_plot(data, description, "Time (Hours)", units)
    
def torch_train_test_split(data, ratio):
    cutoff_idx = int(len(data) * ratio)
    train_y = torch.tensor(data[:cutoff_idx])
    test_y = torch.tensor(data[cutoff_idx:])
    train_x = torch.linspace(0, cutoff_idx, cutoff_idx)
    test_x = torch.linspace(cutoff_idx, len(data), len(data)-cutoff_idx)

    return train_x, train_y, test_x, test_y

# Global variables are bad, m'kay?
def main():
    NH_data = read_xls("xls_data/2011_smd_hourly.xls", "NH")
    data_cutoff = (24 * 8)
    feature_data = np.array(
        [
            NH_data["DEMAND"][:data_cutoff], 
            NH_data["DryBulb"][:data_cutoff], 
            NH_data["DewPnt"][:data_cutoff]
        ]
    )
    feature_description = [
        "Non-PTF Demand",
        "Dry bulb temperature for the weather station",
        "Dew point temperature for the weather station"
    ]
    feature_units = [
        "$/MWh",
        "Temperature (Fahrenheit)",
        "Temperature (Fahrenheit)"
    ]

    #plot_ISO_features(feature_data, feature_description, feature_units)

    for feats in feature_data:
        train_x, train_y, test_x, test_y = torch_train_test_split(
            feats, 
            0.8
        )
    
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_model = gptu.RBFGPModel(train_x, train_y, likelihood)
        optimizer = torch.optim.Adam(
            [
                {"params" : gp_model.parameters()},
            ],
            lr = 0.1
        )
    
        gptu.TorchTrain(train_x, train_y, gp_model, likelihood, optimizer, 200)
        #pred = gptu.TorchTest(test_x, gp_model, likelihood)
        pred = gptu.TorchTest(train_x, gp_model, likelihood)
    
        plt.plot(train_x, train_y)
        #pu.PlotGPPred(test_x, test_y, test_x, pred)
        pu.PlotGPPred(train_x, train_y, train_x, pred)

if __name__ == "__main__":
    main()
