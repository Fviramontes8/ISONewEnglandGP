import numpy as np
import math
import matplotlib.pyplot as plt

import torch
import gpytorch

import PlotUtils as pu
import GPyTorchUtils as gptu
import SignalProcessor as sp

def plot_ISO_features(feature_data, feature_descriptions, feature_units):
    for data, description, units in zip(
        feature_data, 
        feature_descriptions, 
        feature_units
    ): 
        pu.general_plot(data, description, "Time (Hours)", units)
    
def torch_one_hour_data_split(data, offset=11):
    assert(len(data)>48)
    # Inclusive beginning 
    begin=0
    # Exclusive end
    end=24
    offset += 24

    train_x = torch.Tensor(data[begin:end]).view(1, 24)
    train_y = torch.Tensor([data[offset]])

    begin = end
    end += 24
    while(end < (len(data) - 48)):
        train_x = torch.vstack([train_x, torch.Tensor(data[begin:end])])
        train_y = torch.cat([train_y, torch.Tensor([data[begin+offset]])])
        begin = end
        end += 24

    test_x = torch.Tensor(data[begin:end])
    test_y = torch.Tensor([data[begin+offset]])
    return train_x, train_y, test_x, test_y

def torch_one_hour_target_split(data, offset=0):
    begin=0
    end=24
    offset += 24
    train_y = torch.Tensor([data[offset]])

    begin = end
    end += 24
    while(end < (len(data) - 48)):
        train_y = torch.cat([train_y, torch.Tensor([data[begin+offset]])])
        begin = end
        end += 24
    test_y = torch.Tensor([data[begin+offset]])
    return train_y, test_y

def one_hour_prediction(input_data):
    train_x, train_y, test_x, test_y = torch_one_hour_data_split(input_data)
    test_x = test_x.view(1, -1)
    #print(f"Train_x shape: {train_x.shape}")
    #print(f"Train_y shape: {train_y.shape}\n")
    #print(f"Test_x shape: {test_x.shape}")
    #print(f"Test_y shape: {test_y.shape}\n")
    
    '''
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
    '''

class OneDayLinearGP():
    def __init__(self, train_x, train_y_mat, likelihood):
        self.train_x = train_x
        self.train_y = train_y_mat
        self.likelihood = likelihood
        self.models = [
            gptu.LinearGPModel(train_x, i, likelihood) for i in train_y_mat
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
            print("")
    def test(self, test_x):
        self.pred = [
            gptu.TorchTest(test_x, i, self.likelihood).mean.numpy() 
            for i in self.models
        ]

# Global variables are bad, m'kay?
def main():
    NE_data = np.load("data/ISONE_CA_DEMAND.npy")
    data_cutoff = (24 * 12) #192
    feature_data = np.array(
        NE_data[:data_cutoff], 
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
    one_hour_prediction(feature_data)

    train_x, train_y, test_x, test_y = torch_one_hour_data_split(feature_data)
    train2_y, test2_y = torch_one_hour_target_split(feature_data, 11)
    for i in range(1, 24):
        tr_y, te_y = torch_one_hour_target_split(feature_data, i)
        train2_y = torch.vstack([train2_y, tr_y])
        test2_y = torch.vstack([test2_y, te_y])

    #train2_y = torch.transpose(train2_y, 1, 0)
    test2_y = torch.squeeze(test2_y)
    test_x = torch.unsqueeze(test_x, 0)

    #print(f"Train x: {train_x.shape}")
    #print(f"Train y: {train_y.shape}\n")
    #print(f"Test x: {test_x.shape}")
    #print(f"Test y: {test_y.shape}")
    #print(f"train2_y: {train2_y.shape}")
    #print(f"test2_y: {test2_y.shape}\n")


    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = OneDayLinearGP(train_x, train2_y, likelihood)

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
    for i in range(24):
        print(f"{gp_model.pred[i][0]:.1f} {test2_y[i]}")
    local_pred = [i for i in gp_model.pred]

    test_x_plot = [i for i in range(24)]
    plt.plot(test_x_plot, test2_y, label="Actual")
    plt.plot(test_x_plot, local_pred, label="Predicted")
    plt.title("Linear Gaussian Process prediction one day ahead")
    plt.xlabel("Time (hours)")
    plt.ylabel("Load demand (MW)")
    plt.legend()
    plt.show()

    print(f"MAPE: {sp.mape_test(np.array(test2_y), np.array(local_pred)):.3f}")
    #plt.plot(train_x, train_y)
    #pu.PlotGPPred(test_x, test2_y, test_x, local_pred)
    #pu.PlotGPPred(train_x, train_y, train_x, pred)

if __name__ == "__main__":
    main()
