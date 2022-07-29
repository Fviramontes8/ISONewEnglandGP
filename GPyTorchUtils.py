# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:58:52 2020

@author: Frankie
"""

import torch
import gpytorch

import matplotlib.pyplot as plt

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, rank, task_number=2):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks = task_number,
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            ),
            num_tasks = task_number,
            rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(
            mean_x,
            covar_x
        )

class LinearGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class OneDayLinearGP():
    def __init__(self, train_x, train_y_mat, likelihood):
        self.train_x = train_x
        self.train_y = train_y_mat
        self.likelihood = likelihood
        self.models = [
            LinearGPModel(train_x, i, likelihood) for i in train_y_mat
        ]

    def train(self, optimizers, iterations):
        for i, j, k in zip(self.train_y, self.models, optimizers):
            TorchTrain(
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
            TorchTest(test_x, i, self.likelihood) 
            for i in self.models
        ]
        self.lower_sigma = [
            i.confidence_region()[0].item()
            for i in self.pred
        ]
        self.upper_sigma = [
            i.confidence_region()[1].item()
            for i in self.pred
        ]
        self.lower_two_sigma = self.calc_2sigma(
            self.pred, 
            self.lower_sigma
        )
        self.upper_two_sigma = self.calc_2sigma(
            self.pred,
            self.upper_sigma
        )


    def calc_2sigma(self, mean, sigma):
        two_sigma = [
            (mu_and_sigma + (1.96 * (mu_and_sigma - mu.mean.numpy()))).item()
            for mu, mu_and_sigma in zip(mean, sigma)
        ]
        return two_sigma

    def plot_pred(self, test_y, title, xlabel, ylabel):
        x_plot = [i for i in range(len(self.pred))]
        pred_mean = [i.mean.numpy() for i in self.pred]
        plt.fill_between(
            x_plot,
            self.lower_two_sigma,
            self.upper_two_sigma,
            alpha=0.5,
            label="2 Standard deviations"
        )
        plt.fill_between(
            x_plot,
            self.lower_sigma,
            self.upper_sigma,
            alpha=0.5,
            label="1 Standard deviation"
        )
        plt.plot(x_plot, test_y, label="Actual", color="k")
        plt.plot(x_plot, pred_mean, label="Predicted", color="b")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()


class RBFGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(RBFGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def torch_one_hour_data_split(data, offset=11, return_y=True):
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
    if return_y:
        return train_x, train_y, test_x, test_y
    return train_x, test_x

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

def TorchTrain(Xtr, Ytr, GPModel, GPLikelihood, GPOptimizer, TrainingIter, Verbose=False):
    GPModel.train()
    GPLikelihood.train()

    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(
        GPLikelihood,
        GPModel
    )
    if Verbose:
        loss_list = []
        noise_list = []

    for i in range(TrainingIter):
        GPOptimizer.zero_grad()

        #print(f"Xtr shape: {Xtr.shape}")
        #print(f"Ytr shape: {Ytr.shape}")
        output = GPModel(Xtr)
        #print(f"Output shape: {output.mean.shape}")

        loss = -marginal_log_likelihood(output, Ytr)
        if Verbose:
            loss_list.append(loss.detach().numpy())
            noise_list.append(GPModel.likelihood.noise.item())
        #print(f"Loss shape: {loss.shape}")
        loss.backward()

        print("Iter %03d/%03d - Loss: %.3f\tnoise: %.3f" % (
                i + 1, TrainingIter, loss.item(),
                GPModel.likelihood.noise.item()
            )
        )

        GPOptimizer.step()

    if Verbose:
        return loss_list, noise_list

def TorchTrainMultiFeature(Xtr, Ytr, GPModel, GPLikelihood, GPOptimizer, TrainingIter):
    GPModel.train()
    GPLikelihood.train()

    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(
        GPLikelihood,
        GPModel
    )

    for i in range(TrainingIter):
        GPOptimizer.zero_grad()

        output = GPModel(Xtr)

        loss = -marginal_log_likelihood(output, Ytr)
        # Must be loss.backward() if there is a single value for loss
        # Otherwise should be loss.sum().backward() or loss.mean().backward() for multiple values for loss
        #loss.sum().backward()
        loss.mean().backward()

        print("Iter %03d/%03d - Loss: %.3f\tnoise: %.3f" % (
            i + 1, TrainingIter, loss.mean().item(),
            GPModel.likelihood.noise.item()
        ))

        GPOptimizer.step()

def TorchTest(Xtst, GPModel, GPLikelihood):
    GPModel.eval()
    GPLikelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = GPModel(Xtst)
        observed_pred = GPLikelihood(output)

    return observed_pred

"""
Takes a torch tensor and returns lower and upper confidence region for one
standard deviation
"""
def ToStdDev1MT(pred_mean, lower2sigma, upper2sigma):
    lower1sigma = ( (lower2sigma.numpy() - pred_mean) / 1.96) + pred_mean
    upper1sigma = ( (upper2sigma.numpy() - pred_mean) / 1.96) + pred_mean

    return lower1sigma, upper1sigma

"""
Takes a torch tensor and returns lower and upper confidence region for one
standard deviation
"""
def ToStdDev1(pred_mean):
    lower2sigma, upper2sigma = pred_mean.confidence_region()

    lower1sigma = ( (lower2sigma.numpy() - pred_mean.mean.numpy()) / 1.96) + pred_mean.mean.numpy()
    upper1sigma = ( (upper2sigma.numpy() - pred_mean.mean.numpy()) / 1.96) + pred_mean.mean.numpy()

    return lower1sigma, upper1sigma

"""
Returns the percent of data contained with in 1 and 2 standard deviations
YPred should be a torch tensor where YPred.mean.numpy() is a valid method call
YTrue should be a numpy array
"""
def verify_confidence_region(YPred, YTrue):
    y_pred_mean = YPred.mean.numpy()
    y_true = YTrue.numpy()
    assert (len(y_pred_mean) == len(y_true))

    y_lower_sigma, y_upper_sigma = YPred.confidence_region()
    y_lower_sigma1, y_upper_sigma1 = ToStdDev1(YPred)
    y_lower_sigma2, y_upper_sigma2 = y_lower_sigma.numpy(), y_upper_sigma.numpy()

    sigma1_count = 0
    sigma2_count = 0

    for i in range(len(y_pred_mean)):
        if y_lower_sigma1[i] <= y_true[i] and y_upper_sigma1[i] >= y_true[i]:
            sigma1_count += 1

        if y_lower_sigma2[i] <= y_true[i] and y_upper_sigma2[i] >= y_true[i]:
            sigma2_count += 1

    return sigma1_count/len(YTrue), sigma2_count/len(YTrue)

def confirm_confidence_region(ypred, ytrue, upper_sigma, lower_sigma):
    sigma_count = 0

    for i in range(len(ypred)):
        if lower_sigma[i] <= ytrue[i] and upper_sigma[i] >= ytrue[i]:
            sigma_count += 1

    return sigma_count/len(ytrue)
