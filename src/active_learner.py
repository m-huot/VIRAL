# create an active learner that is based on a gaussian process predictor

import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import sympy as sp

from bio_model import *
from sklearn.metrics import roc_auc_score

from gaussian_process import *


def rank_array(fitnesses):
    """
    Optimized function to rank fitness values using NumPy's argsort.

    :param fitnesses: List or NumPy array of fitness values
    :return: NumPy array of ranks
    """
    sorted_indices = np.argsort(fitnesses)
    ranks = np.argsort(sorted_indices)

    return ranks + 1  # Adding 1 to make ranks start from 1


def retrace_indices(selections, size):
    """
    Retrace indices back to the original ensemble.

    Args:
        selections (list of lists): Each sublist contains the indices of points
                                    selected relative to the remaining ensemble at that time.

    Returns:
        list of lists: Each sublist contains the indices of the selected points relative to the original ensemble.
    """
    # Start with a list of indices corresponding to the original ensemble
    original_indices = list(range(size))
    result = []

    for round_indices in selections:
        # Get the indices in the original ensemble for the current selection
        selected_original_indices = [original_indices[i] for i in round_indices]
        result.append(selected_original_indices)

        # Update the remaining indices by removing the selected ones
        original_indices = [
            v for i, v in enumerate(original_indices) if i not in round_indices
        ]

    return result


class Dataset_perso(object):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def get_data(self):
        return self.data_x, self.data_y

    # len
    def __len__(self):
        return len(self.data_x)


class ActiveLearner(object):
    def __init__(
        self,
        train_dataset0,
        dataset,
        strategy="random",
        beta=0.2,
        percent=10,
        biomodel_f=bio_model,
        biomodel_var_f=bio_model_var,
    ):
        """
        Args:


        train_dataset: a Dataset object that contains the training data
        dataset: a Dataset object that contains ALL data
        strategy: the active learning strategy to use
        """

        self.gp = None
        self.train_dataset = train_dataset0
        self.dataset = dataset
        self.strategy = strategy
        self.beta = beta
        self.percent = percent
        self.bio_model = biomodel_f
        self.bio_model_var = biomodel_var_f
        self.training_indices_history = []  # added training history
        # find all points in test_x that are not in train_x
        total_x, total_y = self.dataset.get_data()  # --------------------------
        train_x, train_y = self.train_dataset.get_data()
        test_indices = []
        train0_indexes = []
        for i in range(len(total_x)):
            if np.isin(total_x[i], train_x).all():
                train0_indexes.append(i)
            else:
                test_indices.append(i)
        test_x = total_x[test_indices]
        test_y = total_y[test_indices]
        self.training_indices_history.append(train0_indexes)
        self.test_dataset = Dataset_perso(test_x, test_y)
        print("created new AL object")
        # to np

    def train(self):
        train_x, train_y = self.train_dataset.get_data()
        print("training on ", len(train_x), " points")
        # fing unique points

        self.gp = gp_predictor_sklearn(train_x, train_y)

        self.gp.train_pred()

    def predict(self):
        test_x, _ = self.test_dataset.get_data()
        mean, var = self.gp.predict_pred(test_x)
        return mean, var

    def evaluate(self):
        total_x, total_y = self.dataset.get_data()
        total_y = total_y.detach().numpy()
        mean, var = self.gp.predict_pred(total_x)

        mean = mean.detach().numpy()
        var = var.detach().numpy()
        var = var.reshape(-1, 5)

        # len of total_y
        print("total_y: ", len(total_y))

        train_x, train_y = self.train_dataset.get_data()
        train_y = train_y.detach().numpy()

        pred_fitness_list = self.bio_model(mean)
        label_fitness_list = self.bio_model(total_y)

        pred_fitness_list = np.array(pred_fitness_list)
        label_fitness_list = np.array(label_fitness_list)
        # if nan, raise error
        if np.isnan(pred_fitness_list).any():
            raise ValueError("nan in pred_fitness_list")

        train_fitness_list = self.bio_model(train_y)
        train_fitness_list = np.array(train_fitness_list)

        # find nb of points in pred_fitness_list that are in top 1% of label_fitness_list
        # sort label_fitness_list
        total_label_fitness_list_sorted = np.sort(label_fitness_list)

        # find threshold
        threshold = total_label_fitness_list_sorted[
            int(len(total_label_fitness_list_sorted) * (100 - self.percent) / 100)
        ]
        # find nb of labels above threshold
        nb_labels_above_threshold = 0
        for i in range(len(label_fitness_list)):
            if label_fitness_list[i] >= threshold:
                nb_labels_above_threshold += 1

        # find nb of points in pred_fitness_list that are in top 1% of label_fitness_list
        percent = 0
        for i in range(len(train_fitness_list)):
            if train_fitness_list[i] >= threshold:
                percent += 1
        percent = percent / nb_labels_above_threshold

        labels = label_fitness_list > threshold
        preds = pred_fitness_list

        auc = roc_auc_score(labels, preds)
        return (
            percent,
            auc,
            train_x.var(dim=0).mean().detach().numpy(),
        )

    def acquisition_function(self):
        """
        score potential points to add to the training set
        """
        print("acquisition function with strategy: ", self.strategy)
        test_x, test_y = self.test_dataset.get_data()

        if self.strategy == "random":
            scores = np.random.rand(len(test_x))

        elif self.strategy == "greedy":
            mean, var = self.predict()
            mean = mean.detach().numpy()
            fitnesses = self.bio_model(mean)
            scores = fitnesses

        elif self.strategy == "UCB":
            mean, var = self.predict()
            mean = mean.detach().numpy()
            var = var.detach().numpy()
            var = var.reshape(-1, 5)

            fitnesses = self.bio_model(mean)
            vars = self.bio_model_var(var, mean)
            # shape
            # beta: ,ratio of range of fitnesses to range of vars
            # coeff = (np.max(fitnesses) - np.min(fitnesses)) / (
            #     np.max(np.sqrt(vars)) - np.min(np.sqrt(vars))
            # )
            # ratio of std instead of range
            coeff = fitnesses.std() / np.sqrt(vars).std()
            # top 20 of fitnesses
            scores = fitnesses + np.sqrt(vars) * self.beta * coeff
            # top 20 of scores

        elif self.strategy == "UCB_ranked":
            mean, var = self.predict()
            mean = mean.detach().numpy()
            var = var.detach().numpy()
            var = var.reshape(-1, 5)

            fitnesses = self.bio_model(mean)
            # to array
            fitnesses = np.array(fitnesses)
            vars = self.bio_model_var(var, mean)
            fitnesses_ranked = rank_array(fitnesses)
            vars_ranked = rank_array(vars)
            # shape
            # beta: ,ratio of range of fitnesses to range of vars
            scores = fitnesses_ranked - vars_ranked * self.beta * 0.1

        elif self.strategy == "TS":
            # thomspon sampling
            mean, var = self.predict()
            mean = mean.detach().numpy()
            var = var.detach().numpy()
            var = var.reshape(-1, 5)

            fitnesses = self.bio_model(mean)
            vars = self.bio_model_var(var, mean)

            scores = np.random.normal(fitnesses, np.sqrt(vars))

        return np.array(scores)

    def get_next_points(self, nb_points=1):
        """
        add nb_points to the training set
        """
        test_x, test_y = self.test_dataset.get_data()
        train_x, train_y = self.train_dataset.get_data()

        test_x = test_x.detach().numpy()
        test_y = test_y.detach().numpy()
        mean, var = self.predict()
        mean = mean.detach().numpy()
        var = var.detach().numpy()
        var = var.reshape(-1, 5)
        scores = self.acquisition_function()

        # find the indices beginning with max score
        max_indices = np.argsort(scores)[-nb_points:]
        self.training_indices_history.append(max_indices)  # added training history

        # add the points to the training set

        train_x = np.concatenate((train_x, test_x[max_indices]))
        train_y = np.concatenate((train_y, test_y[max_indices]))
        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        self.train_dataset = Dataset_perso(train_x, train_y)

        # drop the points from the test set
        test_x = np.delete(test_x, max_indices, 0)
        test_y = np.delete(test_y, max_indices, 0)
        test_x = torch.from_numpy(test_x)
        test_y = torch.from_numpy(test_y)
        self.test_dataset = Dataset_perso(test_x, test_y)

    def get_training_indices_history(self):  # added training history
        total_x, total_y = self.dataset.get_data()
        total_y = total_y.detach().numpy()
        return retrace_indices(self.training_indices_history, total_y.shape[0])
