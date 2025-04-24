import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sympy as sp


def load_esm_embeddings(model_name, data="bloom"):
    # Define the mapping between model name and file path
    if data == "bloom":
        path = "../data_bloom/embed_bloom/"
        file_map = {
            "esm1": "embeddings_sequence_esm1v_650m.pt",
            "esm2": "embeddings_sequence_esm2_650m.pt",
            "esm3": "embeddings_sequence_esm3.pt",
            "esm3_coord": "embeddings_withCoordinates_Isolated_WT_down_state_6xf5.pt",
        }
    elif data == "desai":
        path = "../data_desai/embed_desai/"
        file_map = {
            "esm1": "embeddings_sequence_esm1v_650m.pt",
            "esm2": "embeddings_sequence_esm2_650m.pt",
            "esm3": "embeddings_sequence_esm3_only_desai_old.pt",
            "esm3_coord": "embeddings_withCoordinates_Isolated_WT_down_state_6xf5.pt",
        }
    else:
        raise ValueError(f"Invalid data '{data}'. Valid options are: 'bloom', 'desai'")

    # Check if the given model_name exists in the file_map
    if model_name not in file_map:
        raise ValueError(
            f"Invalid model name '{model_name}'. Valid options are: {list(file_map.keys())}"
        )

    # Load the embeddings from the corresponding file

    file_path = path + file_map[model_name]

    # List to store the loaded embeddings
    embeddings_list = []

    # Iterate over files in the folder

    embeddings = torch.load(file_path)  # Load the tensor
    # make sure tensor was loaded correctly
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError(f"Expected tensor, got: {type(embeddings)}")

    # Convert to list of arrays (if needed)
    if isinstance(embeddings, torch.Tensor):
        embeddings_list.append(embeddings)  # Keep as tensor, convert later if needed

    # Combine all tensors into one
    combined_embeddings = torch.cat(embeddings_list, dim=0)
    print("loaded embeddings of shape", combined_embeddings.shape)

    # to numpy array
    combined_embeddings = combined_embeddings.numpy()

    # normalize
    combined_embeddings = (
        combined_embeddings - np.mean(combined_embeddings, axis=0)
    ) / np.std(combined_embeddings, axis=0)

    return combined_embeddings
