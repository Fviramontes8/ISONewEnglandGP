from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (
    CosineKernel,
    LinearKernel,
    PeriodicKernel,
    ProductKernel,
    RBFKernel,
    ScaleKernel,
)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.settings import fast_pred_var

from torch import no_grad, Tensor
from torch.optim import Adam


class LinearGPModel(ExactGP):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: GaussianLikelihood,
    ):
        super(LinearGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(LinearKernel())

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class RBFGPModel(ExactGP):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: GaussianLikelihood,
    ):
        super(RBFGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class CosineGPModel(ExactGP):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: GaussianLikelihood,
    ):
        super(CosineGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(CosineKernel())

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class CustomGPModel(ExactGP):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: GaussianLikelihood,
    ):
        super(CustomGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(PeriodicKernel())

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GaussianProcess:
    def __init__(
        self,
        gpmodel: LinearGPModel or RBFGPModel or CosineGPModel or CustomGPModel,
        likelihood: GaussianLikelihood,
        optimizer: Adam,
        train_x: Tensor,
        train_y: Tensor,
    ):
        self.model = gpmodel
        self.likelihood = likelihood
        self.optimizer = optimizer
        self.train_x = train_x
        self.train_y = train_y

    def __call__(self, x_data: Tensor):
        self.model.eval()
        self.likelihood.eval()

        with no_grad(), fast_pred_var():
            pred = self.likelihood(self.model(x_data))

        return pred

    def train(self, epoch: int, debug: bool = False):
        if debug:
            print(f"x train shape: {self.train_x.shape}")
            print(f"y train shape: {self.train_y.shape}")
        self.model.train()
        self.likelihood.train()
        marginal_log_likelihood = ExactMarginalLogLikelihood(
            self.likelihood, self.model
        )

        loss_list = []
        noise_list = []

        for i in range(epoch):
            self.optimizer.zero_grad()

            output = self.model(self.train_x)
            loss = -marginal_log_likelihood(output, self.train_y)

            # if debug:
            #     print(f"Loss shape: {loss.shape}")

            loss_list.append(loss.detach().numpy())
            noise_list.append(self.model.likelihood.noise.item())
            loss.backward()

            if ((i + 1) % 10) == 0:
                print(
                    f"Iter {(i+1):04d}/{epoch:04d} -"
                    f" Loss: {loss.item():.3f}"
                    f" Noise: {self.model.likelihood.noise.item():.3f}"
                )

            self.optimizer.step()

        return loss_list, noise_list


def train_model(
    xtr: Tensor,
    ytr: Tensor,
    gpmodel: LinearGPModel,
    likelihood: GaussianLikelihood,
    optimizer,
    epoch,
    debug=False,
):
    if debug:
        print(f"xtr shape: {xtr.shape}")
        print(f"ytr shape: {ytr.shape}")
    gpmodel.train()
    likelihood.train()
    marginal_log_likelihood = ExactMarginalLogLikelihood(likelihood, gpmodel)

    loss_list = []
    noise_list = []

    for i in range(epoch):
        optimizer.zero_grad()

        output = gpmodel(xtr)
        loss = -marginal_log_likelihood(output, ytr)

        if debug:
            print(f"Loss shape: {loss.shape}")

        loss_list.append(loss.detach().numpy())
        noise_list.append(gpmodel.likelihood.noise.item())
        loss.backward()

        if (i + 1) % 10:
            print(
                f"Iter {(i+1):04d}/{epoch:04d} -"
                f" Loss: {loss.item():.3f}"
                f" Noise: {gpmodel.likelihood.noise.item():.3f}"
            )

        optimizer.step()

    return loss_list, noise_list
