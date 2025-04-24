import torch
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic


class gp_predictor_sklearn:
    def __init__(self, train_x, train_y):
        train_x = train_x.detach().numpy()
        train_y = train_y.detach().numpy()

        kernel = RationalQuadratic()

        self.model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=0, random_state=0
        )

        self.train_x = train_x
        self.train_y = train_y

    def train_pred(self):
        # Fit the GP model to the training data
        self.model.fit(self.train_x, self.train_y)
        print("Learned kernel: %s" % self.model.kernel_)

    def predict_pred(self, test_x):
        # tenspr to array
        test_x = test_x.detach().numpy()
        # Make predictions on the test data
        mean_predictions, std_predictions = self.model.predict(test_x, return_std=True)
        var_predictions = std_predictions**2

        # Clip predictions to a specific range
        mean_predictions = np.clip(mean_predictions, -10, 10)
        var_predictions = np.clip(
            var_predictions, 1e-10, 10
        )  # Clip variance to avoid extreme values

        mean_predictions = torch.from_numpy(mean_predictions)
        var_predictions = torch.from_numpy(var_predictions)
        return mean_predictions, var_predictions
