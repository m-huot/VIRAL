import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split

sys.path.append("../src")
from utils import *
from gaussian_process import *
from active_learner import *
from hist_al_utils import *


def main(EMBED):
    # Load and preprocess data
    df, train_x, train_y, targets = load_and_preprocess_data(EMBED)
    dataset = Dataset_perso(train_x, train_y)
    site_rbd_list = np.unique(df["site_SARS2"].values)

    FOLDER = f"../script_results/hist_al_bloom_{EMBED}"
    # create folder if it does not exist
    if not os.path.exists(FOLDER):
        print(f"Creating folder {FOLDER}")
        os.makedirs(FOLDER)
    NB_POINTS = 50  # 50
    NB_ROUNDS = 10  # 10
    NB_RUNS = 10
    NB_0 = 165

    # Run active learning and save results
    get_hist_bloom(
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
    )

    # Generate points list for plotting
    points_list = [NB_0 + i * NB_POINTS for i in range(NB_ROUNDS + 1)]

    # Plot and save results
    plot_results(FOLDER, points_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Active Learning Experiment")
    parser.add_argument(
        "--embed",
        type=str,
        default="esm3_coord",
        help="Embedding type (e.g., esm1, esm3, esm3_coord)",
    )
    args = parser.parse_args()

    main(args.embed)
