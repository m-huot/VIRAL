import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
import random
import csv

# Import required modules from src
sys.path.append("../src")
from utils import *
from gaussian_process import *
from active_learner import *
from bio_model import *


def load_and_preprocess_data(EMBED, df="bloom"):
    df_code = df
    # Load datasets
    if df_code == "bloom":
        df_kd = pd.read_csv("../data_bloom/kd_bloom/df_bloom_processed.csv")
        # rename columns site as site_SARS2
        df_kd.rename(columns={"site": "site_SARS2"}, inplace=True)
        train_data = load_esm_embeddings(EMBED)

    elif df_code == "desai":
        df_kd = pd.read_csv("../data_desai/kd_desai/df_desai_processed.csv")
        train_data = load_esm_embeddings(EMBED, data="desai")

    targets = [
        "delta_log_kd_ACE2",
        "delta_log_kd_LY-CoV016",
        "delta_log_kd_REGN10987",
        "delta_log_kd_LY-CoV555",
        "delta_log_kd_S309",
    ]

    train_x = torch.tensor(train_data)
    train_y = torch.tensor(df_kd[targets].values)

    # Remove rows with NaN
    indexes_nan = np.unique(np.where(np.isnan(train_y))[0])
    non_nan_indexes = np.setdiff1d(np.arange(train_y.shape[0]), indexes_nan)

    train_x = train_x[non_nan_indexes]
    train_y = train_y[non_nan_indexes]
    df = df_kd.drop(indexes_nan).reset_index(drop=True)

    df[EMBED] = train_x.tolist()

    # remove rows without embedding in esm3 coord
    if df_code == "bloom":
        indexNames = []
        for i in range(len(df)):
            if df["site_SARS2"][i] in [331, 332, 333, 527, 528, 529, 530, 531]:
                indexNames.append(i)

        # Delete these row indexes from dataFrame
        df.drop(indexNames, inplace=True)
        df = df.reset_index(drop=True)

        train_x = np.delete(train_x, indexNames, axis=0)
        train_y = np.delete(train_y, indexNames, axis=0)

    return df, train_x, train_y, targets


def run_active_learner(
    strategy,
    dataset_0,
    dataset,
    NB_POINTS,
    NB_ROUNDS,
    biomodel="fitness",
):
    if biomodel == "fitness":
        active_learner = ActiveLearner(
            dataset_0, dataset, strategy=strategy, percent=10
        )

    elif biomodel == "direct":
        active_learner = ActiveLearner(
            dataset_0,
            dataset,
            strategy=strategy,
            percent=10,
            biomodel_f=bio_model_fitness,
            biomodel_var_f=bio_model_var_fitness,
        )
    else:
        raise ValueError("Invalid biomodel")
    active_learner.train()
    p, r2, var = active_learner.evaluate()

    r2_list, p_list, var_list = [r2], [p], [var]

    for _ in range(NB_ROUNDS):
        active_learner.get_next_points(nb_points=NB_POINTS)
        active_learner.train()
        p, r2, var = active_learner.evaluate()
        r2_list.append(r2)
        p_list.append(p)
        var_list.append(var)
        print(f"Strategy: {strategy}, AUC: {r2}, P: {p}, Var: {var}")

    return np.array(r2_list), np.array(p_list), np.array(var_list), active_learner


def compute_mean_std(matrices, keys):
    results = {}
    for key in keys:
        results[f"{key}_mean"] = np.mean(matrices[key], axis=0)
        results[f"{key}_std"] = np.std(matrices[key], axis=0)
    return results


def get_hist_bloom(
    NB_POINTS,
    NB_ROUNDS,
    NB_RUNS,
    NB_0,
    site_rbd_list,
    dataset,
    FOLDER,
    df,
    EMBED,
    biomodel="fitness",
):
    strategies = ["random", "greedy", "UCB", "UCB_ranked"]
    matrices = {
        s: {
            "r2": np.zeros((NB_RUNS, NB_ROUNDS + 1)),
            "p": np.zeros((NB_RUNS, NB_ROUNDS + 1)),
            "var": np.zeros((NB_RUNS, NB_ROUNDS + 1)),
        }
        for s in strategies
    }

    points_list = [NB_0 + i * NB_POINTS for i in range(NB_ROUNDS + 1)]

    for run in range(NB_RUNS):
        train_0_indexes = [
            df[df["site_SARS2"] == i].index[
                np.random.randint(0, df[df["site_SARS2"] == i].shape[0])
            ]
            for i in site_rbd_list
        ]
        if NB_0 < len(train_0_indexes):
            print("Resampling train_0_indexes to match NB_0")
            train_0_indexes = np.random.choice(
                train_0_indexes, NB_0, replace=False
            ).tolist()

        train_0_x = torch.stack([dataset.data_x[i] for i in train_0_indexes])
        train_0_y = torch.stack([dataset.data_y[i] for i in train_0_indexes])
        dataset_0 = Dataset_perso(train_0_x, train_0_y)

        active_learners = {}
        # Start plot
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))  # Create 3 subplots vertically

        for strategy in strategies:
            r2_list, p_list, var_list, active_learner = run_active_learner(
                strategy=strategy,
                dataset_0=dataset_0,
                dataset=dataset,
                NB_POINTS=NB_POINTS,
                NB_ROUNDS=NB_ROUNDS,
                biomodel=biomodel,
            )

            train_data_u, _ = active_learner.train_dataset.get_data()
            if (strategy == "UCB" or strategy == "greedy") and EMBED == "esm3_coord":
                train_data_u_indexes = []
                for j in range(len(df)):
                    for i in range(len(train_data_u)):
                        if df[EMBED][j] == train_data_u[i].tolist():
                            train_data_u_indexes.append(j)

                print("indexes checked by embedding", train_data_u_indexes)

                training_set = df.loc[train_data_u_indexes]
                # save training set in csv with name training_set_+strategy+"_run_"+run
                filename = f"training_set_{strategy}_run_{run}.csv"

                # Save the training set to a CSV file
                training_set.to_csv(FOLDER + "/" + filename, index=False)

                hist_indexes = active_learner.get_training_indices_history()
                # save as a npy with the right name
                # Save the history as a .npy file
                print("hist_indexes", hist_indexes)
                filename = f"{FOLDER}/training_indices_history_{strategy}_run_{run}.csv"

                # Save list of lists to a CSV file
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(hist_indexes)

                print(f"List of lists saved as CSV to {filename}")

            matrices[strategy]["r2"][run, :] = r2_list
            matrices[strategy]["p"][run, :] = p_list
            matrices[strategy]["var"][run, :] = var_list
            active_learners[strategy] = active_learner

            # Plot r2 on the first subplot
            axs[0].plot(points_list, r2_list, label=strategy)

            # Plot p on the second subplot
            axs[1].plot(points_list, p_list, label=strategy)

            # Plot var on the third subplot
            axs[2].plot(points_list, var_list, label=strategy)

        # Labels and titles for the subplots
        axs[0].set_xlabel("Number of points", fontsize=12)
        axs[0].set_ylabel("AUC", fontsize=12)

        axs[1].set_xlabel("Number of points", fontsize=12)
        axs[1].set_ylabel("P", fontsize=12)

        axs[2].set_xlabel("Number of points", fontsize=12)
        axs[2].set_ylabel("Variance", fontsize=12)

        # Add legends to each subplot
        for ax in axs:
            ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save and show the plot
        plt.savefig(FOLDER + f"/plot_{run}.png")
        plt.show()

    result_r2 = compute_mean_std({s: matrices[s]["r2"] for s in strategies}, strategies)
    result_p = compute_mean_std({s: matrices[s]["p"] for s in strategies}, strategies)
    result_var = compute_mean_std(
        {s: matrices[s]["var"] for s in strategies}, strategies
    )

    for key, values in result_p.items():
        np.save(FOLDER + f"/{key}" + "_p.npy", values)
    for key, values in result_r2.items():
        np.save(FOLDER + f"/{key}" + "_r.npy", values)
    for key, values in result_var.items():
        np.save(FOLDER + f"/{key}" + "_var.npy", values)


def plot_results(FOLDER, points_list):
    r2_list_u_mean = np.load(FOLDER + "/UCB_mean_r.npy")
    r2_list_u_std = np.load(FOLDER + "/UCB_std_r.npy")
    r2_list_g_mean = np.load(FOLDER + "/greedy_mean_r.npy")
    r2_list_g_std = np.load(FOLDER + "/greedy_std_r.npy")
    r2_list_r_mean = np.load(FOLDER + "/random_mean_r.npy")
    r2_list_r_std = np.load(FOLDER + "/random_std_r.npy")
    r2_list_ur_mean = np.load(FOLDER + "/UCB_ranked_mean_r.npy")
    r2_list_ur_std = np.load(FOLDER + "/UCB_ranked_std_r.npy")

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        points_list, r2_list_u_mean, yerr=r2_list_u_std, fmt="o", label="UCB", color="g"
    )
    plt.errorbar(
        points_list,
        r2_list_g_mean,
        yerr=r2_list_g_std,
        fmt="o",
        label="greedy",
        color="b",
    )
    plt.errorbar(
        points_list,
        r2_list_r_mean,
        yerr=r2_list_r_std,
        fmt="o",
        label="random",
        color="r",
    )

    plt.errorbar(
        points_list,
        r2_list_ur_mean,
        yerr=r2_list_ur_std,
        fmt="o",
        label="UCB_ranked",
        color="purple",
    )

    plt.xlabel("Number of points", fontsize=12)
    plt.ylabel("AUC", fontsize=12)
    plt.title("AUC vs Number of points", fontsize=14)
    plt.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER + "/spearman_vs_points.png")
    plt.show()

    p_list_u_mean = np.load(FOLDER + "/UCB_mean_p.npy")
    p_list_u_std = np.load(FOLDER + "/UCB_std_p.npy")
    p_list_g_mean = np.load(FOLDER + "/greedy_mean_p.npy")
    p_list_g_std = np.load(FOLDER + "/greedy_std_p.npy")
    p_list_r_mean = np.load(FOLDER + "/random_mean_p.npy")
    p_list_r_std = np.load(FOLDER + "/random_std_p.npy")
    p_list_ur_mean = np.load(FOLDER + "/UCB_ranked_mean_p.npy")
    p_list_ur_std = np.load(FOLDER + "/UCB_ranked_std_p.npy")

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        points_list, p_list_u_mean, yerr=p_list_u_std, fmt="o", label="UCB", color="g"
    )
    plt.errorbar(
        points_list,
        p_list_g_mean,
        yerr=p_list_g_std,
        fmt="o",
        label="greedy",
        color="b",
    )
    plt.errorbar(
        points_list,
        p_list_r_mean,
        yerr=p_list_r_std,
        fmt="o",
        label="random",
        color="r",
    )

    plt.errorbar(
        points_list,
        p_list_ur_mean,
        yerr=p_list_ur_std,
        fmt="o",
        label="UCB_ranked",
        color="purple",
    )

    plt.xlabel("Number of points", fontsize=12)
    plt.ylabel("P", fontsize=12)
    plt.title("P vs Number of points", fontsize=14)
    plt.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER + "/p_vs_points.png")
    plt.show()

    var_list_u_mean = np.load(FOLDER + "/UCB_mean_var.npy")
    var_list_u_std = np.load(FOLDER + "/UCB_std_var.npy")
    var_list_g_mean = np.load(FOLDER + "/greedy_mean_var.npy")
    var_list_g_std = np.load(FOLDER + "/greedy_std_var.npy")
    var_list_r_mean = np.load(FOLDER + "/random_mean_var.npy")
    var_list_r_std = np.load(FOLDER + "/random_std_var.npy")
    var_list_ur_mean = np.load(FOLDER + "/UCB_ranked_mean_var.npy")
    var_list_ur_std = np.load(FOLDER + "/UCB_ranked_std_var.npy")

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        points_list,
        var_list_u_mean,
        yerr=var_list_u_std,
        fmt="o",
        label="UCB",
        color="g",
    )
    plt.errorbar(
        points_list,
        var_list_g_mean,
        yerr=var_list_g_std,
        fmt="o",
        label="greedy",
        color="b",
    )
    plt.errorbar(
        points_list,
        var_list_r_mean,
        yerr=var_list_r_std,
        fmt="o",
        label="random",
        color="r",
    )

    plt.errorbar(
        points_list,
        var_list_ur_mean,
        yerr=var_list_ur_std,
        fmt="o",
        label="UCB_ranked",
        color="purple",
    )

    plt.xlabel("Number of points", fontsize=12)
    plt.ylabel("Variance", fontsize=12)
    plt.title("Variance vs Number of points", fontsize=14)
    plt.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER + "/variance_vs_points.png")
    plt.show()


def get_hist_desai(
    NB_POINTS,
    NB_ROUNDS,
    NB_RUNS,
    NB_0,
    site_rbd_list,
    dataset,
    FOLDER,
    df,
    EMBED,
    biomodel="fitness",
):
    strategies = ["random", "greedy", "UCB", "UCB_ranked"]
    matrices = {
        s: {
            "r2": np.zeros((NB_RUNS, NB_ROUNDS + 1)),
            "p": np.zeros((NB_RUNS, NB_ROUNDS + 1)),
            "var": np.zeros((NB_RUNS, NB_ROUNDS + 1)),
        }
        for s in strategies
    }

    points_list = [NB_0 + i * NB_POINTS for i in range(NB_ROUNDS + 1)]

    for run in range(NB_RUNS):
        WT = "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST"

        # in mutant_sequence of df, find indexes of raws being 0 or 1 mutation away from WT
        def mutation_count(seq1, seq2):
            return sum(1 for a, b in zip(seq1, seq2) if a != b)

        # Filter indexes where the mutant sequence is at least 2 mutations away from WT
        at_least_two_mutations_indexes = df[
            df["mutant_sequence"].apply(lambda seq: mutation_count(seq, WT) <= 2)
        ].index

        # Sample N_0 indexes randomly from the filtered indexes
        if len(at_least_two_mutations_indexes) >= NB_0:
            train_0_indexes = random.sample(list(at_least_two_mutations_indexes), NB_0)
        else:
            # If there are fewer than N_0 samples, take all available
            train_0_indexes = list(at_least_two_mutations_indexes)

        train_0_x = torch.stack([dataset.data_x[i] for i in train_0_indexes])
        train_0_y = torch.stack([dataset.data_y[i] for i in train_0_indexes])
        dataset_0 = Dataset_perso(train_0_x, train_0_y)

        active_learners = {}
        # Start plot
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))  # Create 3 subplots vertically

        for strategy in strategies:
            r2_list, p_list, var_list, active_learner = run_active_learner(
                strategy=strategy,
                dataset_0=dataset_0,
                dataset=dataset,
                NB_POINTS=NB_POINTS,
                NB_ROUNDS=NB_ROUNDS,
                biomodel=biomodel,
            )
            train_data_u, _ = active_learner.train_dataset.get_data()

            if (strategy == "UCB" or strategy == "greedy") and EMBED == "esm3_coord":
                train_data_u_indexes = []
                for j in range(len(df)):
                    for i in range(len(train_data_u)):
                        if df[EMBED][j] == train_data_u[i].tolist():
                            train_data_u_indexes.append(j)
                training_set = df.loc[train_data_u_indexes]
                # save training set in csv with name training_set_+strategy+"_run_"+run
                filename = f"training_set_{strategy}_run_{run}.csv"

                # Save the training set to a CSV file
                training_set.to_csv(FOLDER + "/" + filename, index=False)

                hist_indexes = active_learner.get_training_indices_history()
                # save as a npy with the right name
                # Save the history as a .npy file
                print("hist_indexes", hist_indexes)
                filename = f"{FOLDER}/training_indices_history_{strategy}_run_{run}.csv"

                # Save list of lists to a CSV file
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(hist_indexes)

                print(f"List of lists saved as CSV to {filename}")

            matrices[strategy]["r2"][run, :] = r2_list
            matrices[strategy]["p"][run, :] = p_list
            matrices[strategy]["var"][run, :] = var_list
            active_learners[strategy] = active_learner

            # Plot r2 on the first subplot
            axs[0].plot(points_list, r2_list, label=strategy)

            # Plot p on the second subplot
            axs[1].plot(points_list, p_list, label=strategy)

            # Plot var on the third subplot
            axs[2].plot(points_list, var_list, label=strategy)

        # Labels and titles for the subplots
        axs[0].set_xlabel("Number of points", fontsize=12)
        axs[0].set_ylabel("AUC", fontsize=12)

        axs[1].set_xlabel("Number of points", fontsize=12)
        axs[1].set_ylabel("P", fontsize=12)

        axs[2].set_xlabel("Number of points", fontsize=12)
        axs[2].set_ylabel("Variance", fontsize=12)

        # Add legends to each subplot
        for ax in axs:
            ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save and show the plot
        plt.savefig(FOLDER + f"/plot_{run}.png")
        plt.show()

    result_r2 = compute_mean_std({s: matrices[s]["r2"] for s in strategies}, strategies)
    result_p = compute_mean_std({s: matrices[s]["p"] for s in strategies}, strategies)
    result_var = compute_mean_std(
        {s: matrices[s]["var"] for s in strategies}, strategies
    )

    for key, values in result_p.items():
        np.save(FOLDER + f"/{key}" + "_p.npy", values)
    for key, values in result_r2.items():
        np.save(FOLDER + f"/{key}" + "_r.npy", values)
    for key, values in result_var.items():
        np.save(FOLDER + f"/{key}" + "_var.npy", values)
