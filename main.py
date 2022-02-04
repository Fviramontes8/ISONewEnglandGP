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
    
def torch_one_hour_data_split(data):
    assert(len(data)>35)
    # Inclusive beginning
    begin=0
    # Exclusive end
    end=24

    train_x = torch.Tensor(data[begin:end]).view(1, 24)
    train_y = torch.Tensor([data[36]])

    begin = end
    end += 24
    while(end < (len(data) - 48)):
        train_x = torch.vstack([train_x, torch.Tensor(data[begin:end])])
        train_y = torch.cat([train_y, torch.Tensor([data[begin+36]])])
        begin = end
        end += 24

    test_x = torch.Tensor(data[begin:end])
    test_y = torch.Tensor([data[begin+36]])
    return train_x, train_y, test_x, test_y

def torch_one_day_data_split(data):
    assert(len(data)>35)
    # Inclusive beginning
    begin=0
    # Exclusive end
    end=24

    train_x = torch.Tensor(data[begin:end]).view(1, 24)
    train_y = torch.Tensor([data[end:end+24]])

    begin = end
    end += 24
    while(end < (len(data) - 48)):
        train_x = torch.vstack([train_x, torch.Tensor(data[begin:end])])
        train_y = torch.cat([train_y, torch.Tensor([data[end:end+24]])])
        begin = end
        end += 24

    test_x = torch.Tensor([data[begin:end]])
    test_y = torch.Tensor([data[end:end+24]])
    return train_x, train_y, test_x, test_y

def one_hour_prediction(input_data):
    train_x, train_y, test_x, test_y = torch_one_hour_data_split(input_data)
    test_x = test_x.view(1, -1)
    #print(f"Train_x shape: {train_x.shape}")
    #print(f"Train_y shape: {train_y.shape}")
    #print(f"Test_x shape: {test_x.shape}")
    #print(f"Test_y shape: {test_y.shape}")
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = gptu.LinearGPModel(train_x, train_y, likelihood)
    optimizer = torch.optim.Adam(
        [
            {"params" : gp_model.parameters()},
        ],
        lr = 0.1
    )
    
    train_loss = gptu.TorchTrain(train_x, train_y, gp_model, likelihood, optimizer, 50, True)
    pu.general_plot(train_loss, "Loss over time")
    pred = gptu.TorchTest(test_x, gp_model, likelihood)
    print(f"{test_y.item()}")
    print(f"{pred.mean.numpy()}")

class OneDayLinearGP():
    def __init__(self, train_x, train_y, likelihood):
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood
        self.models = [
            gptu.LinearGPModel(train_x, i, likelihood) for i in train_y
        ]

    def train(self, optimizers, iterations):
        for i, j, k in zip(self.train_y, self.models, optimizers):
            gptu.TorchTrain(
                self.train_x, 
                i, 
                j, 
                self.likelihood, 
                k, 
                iterations
            )
    def test(self, test_x):
        self.pred = [
            gptu.TorchTest(test_x, i, self.likelihood).mean.numpy() 
            for i in self.models
        ]

# Global variables are bad, m'kay?
def main():
    NE_data = read_xls("xls_data/2011_smd_hourly.xls", "ISONE CA")
    data_cutoff = (24 * 12) #192
    feature_data = np.array(
        NE_data["DEMAND"][:data_cutoff], 
    )
    feature_description = "Non-PTF Load Demand" 
    feature_units = "MW"

    '''
    pu.general_plot(
        feature_data, 
        feature_description, 
        "Time (hours)", 
        feature_units
    )
    '''

    print(f"Data length: {len(feature_data)}")
    #one_hour_prediction(feature_data)

    train_x, train_y, test_x, test_y = torch_one_day_data_split(feature_data)
    #print(f"Train x: {train_x.shape}\n")
    #print(f"Train y: {train_y.shape}\n")
    #print(f"Test x: {test_x.shape}")
    #print(f"Test y: {test_y.shape}")

    print(f"Train x: {train_x}\n")
    print(f"Train y: {train_y}\n")

    '''
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.LinearKernel() + gpytorch.kernels.RBFKernel()
    )
    gp_model = OneDayLinearGP(train_x, train_y, likelihood)
    optimizers = [
        torch.optim.Adam(
            [
                {"params" : i.parameters(),}
            ],
            lr = 0.1
        )
        for i in gp_model.models
    ]
    gp_model.train(optimizers, 10)
    gp_model.test(test_x)
    print(f"{gp_model.pred}")
    '''
    
    #gptu.TorchTrain(train_x, train_y, gp_model, likelihood, optimizer, 10)
    #pred = gptu.TorchTest(test_x, gp_model, likelihood)
    #print(f"{test_y.numpy()}")
    #print(f"{pred.mean.numpy()}")

    #plt.plot(train_x, train_y)
    #pu.PlotGPPred(test_x, test_y, test_x, pred)
    #pu.PlotGPPred(train_x, train_y, train_x, pred)

if __name__ == "__main__":
    main()
