import os
import sys
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from tqdm import tqdm
import argparse

import torch
from sklearn.model_selection import train_test_split

# Import required modules from src
sys.path.append("../src")
from utils import *
from gaussian_process import *
from active_learner import *
from hist_al_utils import *


from gaussian_process import gp_predictor_sklearn


def run_benchmark(nb_rounds):
    percent = 10
    embeds = ["esm1", "esm2", "esm3", "esm3_coord"]
    predictors = [
        gp_predictor_sklearn,
    ]
    dilution_bloom = [0.2, 0.4, 0.6, 0.8, 1, 2, 3]
    dilution_desai = [20, 40, 80, 120, 160]

    results = {}

    # Benchmark for Bloom dataset
    for embed in embeds:
        print(f"Processing Bloom dataset with {embed}...")
        df, train_x, train_y, targets = load_and_preprocess_data(embed)
        dataset = Dataset_perso(train_x, train_y)
        site_rbd_list = np.unique(df["site_SARS2"].values)
        total_x, total_y = dataset.get_data()

        for predictor_cls in predictors:
            print(f"Predictor: {predictor_cls.__name__}")
            mean_auc_dilution, std_auc_dilution = [], []
            mean_spearman_dilution, std_spearman_dilution = [], []

            for d in dilution_bloom:
                auc_scores, spearman_scores = [], []
                for j in range(nb_rounds):
                    np.random.seed(j)
                    total_x, total_y = dataset.get_data()

                    if d <= 1:
                        train_0_indexes = [
                            df[df["site_SARS2"] == i].index[
                                np.random.randint(0, df[df["site_SARS2"] == i].shape[0])
                            ]
                            for i in site_rbd_list
                        ]
                        train_0_indexes = np.random.choice(
                            train_0_indexes,
                            int(d * len(train_0_indexes)),
                            replace=False,
                        )
                    else:
                        train_0_indexes = []
                        for i in site_rbd_list:
                            train_0_indexes.extend(
                                np.random.choice(
                                    df[df["site_SARS2"] == i].index,
                                    int(d),
                                    replace=False,
                                )
                            )

                    # Convert train_0_indexes to a set for faster operations
                    train_0_indexes = set(train_0_indexes)

                    # Create the training data
                    train_x = total_x[list(train_0_indexes)]
                    train_y = total_y[list(train_0_indexes)]

                    # Identify test indexes
                    all_indexes = set(range(len(total_x)))
                    test_indexes = all_indexes - train_0_indexes

                    # Convert test_indexes to a list and create test data
                    test_indexes = list(test_indexes)
                    test_x = total_x[test_indexes]
                    test_y = total_y[test_indexes]

                    predictor = predictor_cls(train_x, train_y)
                    predictor.train_pred()

                    mean, var = predictor.predict_pred(test_x)
                    mean = mean.detach().numpy()
                    pred_fitness_list = bio_model(mean)
                    label_fitness_list = bio_model(test_y)

                    threshold = np.sort(label_fitness_list)[
                        int(len(label_fitness_list) * (100 - percent) / 100)
                    ]
                    labels = label_fitness_list > threshold
                    auc = roc_auc_score(labels, pred_fitness_list)
                    spearman_corr, _ = spearmanr(label_fitness_list, pred_fitness_list)

                    auc_scores.append(auc)
                    spearman_scores.append(spearman_corr)

                mean_auc_dilution.append(np.mean(auc_scores))
                std_auc_dilution.append(np.std(auc_scores))
                mean_spearman_dilution.append(np.mean(spearman_scores))
                std_spearman_dilution.append(np.std(spearman_scores))

            results[(embed, predictor_cls.__name__)] = {
                "mean_auc": mean_auc_dilution,
                "std_auc": std_auc_dilution,
                "mean_spearman": mean_spearman_dilution,
                "std_spearman": std_spearman_dilution,
            }
    with open("../script_results/few_shot_benchmark_bloom.pkl", "wb") as f:
        pickle.dump(results, f)

    # Benchmark for Desai dataset
    for embed in embeds:
        print(f"Processing Desai dataset with {embed}...")
        df, train_x, train_y, targets = load_and_preprocess_data(embed, "desai")
        dataset = Dataset_perso(train_x, train_y)
        total_x, total_y = dataset.get_data()

        for predictor_cls in predictors:
            print(f"Predictor: {predictor_cls.__name__}")
            mean_auc_dilution, std_auc_dilution = [], []
            mean_spearman_dilution, std_spearman_dilution = [], []

            for d in dilution_desai:
                auc_scores, spearman_scores = [], []
                for j in range(nb_rounds):
                    np.random.seed(j)
                    total_x, total_y = dataset.get_data()

                    WT = "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST"

                    def mutation_count(seq1, seq2):
                        return sum(1 for a, b in zip(seq1, seq2) if a != b)

                    indexes = df.index

                    if len(indexes) >= d:
                        train_0_indexes = random.sample(list(indexes), d)

                    # Convert train_0_indexes to a set for faster operations
                    train_0_indexes = set(train_0_indexes)

                    # Create the training data
                    train_x = total_x[list(train_0_indexes)]
                    train_y = total_y[list(train_0_indexes)]

                    # Identify test indexes
                    all_indexes = set(range(len(total_x)))
                    test_indexes = all_indexes - train_0_indexes

                    # Convert test_indexes to a list and create test data
                    test_indexes = list(test_indexes)
                    test_x = total_x[test_indexes]
                    test_y = total_y[test_indexes]

                    predictor = predictor_cls(train_x, train_y)
                    predictor.train_pred()

                    mean, var = predictor.predict_pred(test_x)
                    mean = mean.detach().numpy()
                    pred_fitness_list = bio_model(mean)
                    label_fitness_list = bio_model(test_y)

                    threshold = np.sort(label_fitness_list)[
                        int(len(label_fitness_list) * (100 - percent) / 100)
                    ]
                    labels = label_fitness_list > threshold
                    auc = roc_auc_score(labels, pred_fitness_list)
                    spearman_corr, _ = spearmanr(label_fitness_list, pred_fitness_list)

                    auc_scores.append(auc)
                    spearman_scores.append(spearman_corr)

                mean_auc_dilution.append(np.mean(auc_scores))
                std_auc_dilution.append(np.std(auc_scores))
                mean_spearman_dilution.append(np.mean(spearman_scores))
                std_spearman_dilution.append(np.std(spearman_scores))

            results[(embed, predictor_cls.__name__)] = {
                "mean_auc": mean_auc_dilution,
                "std_auc": std_auc_dilution,
                "mean_spearman": mean_spearman_dilution,
                "std_spearman": std_spearman_dilution,
            }

    with open("../script_results/few_shot_benchmark_desai.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run few-shot benchmark")
    parser.add_argument(
        "--nb_rounds", type=int, default=10, help="Number of benchmark rounds"
    )
    args = parser.parse_args()

    run_benchmark(args.nb_rounds)
