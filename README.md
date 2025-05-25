# Overview of VIRAL

Machine-learning workflow that **identifies high-fitness viral protein variants with up to 5× fewer wet-lab measurements**.  
The code base couples Bayesian active learning, Gaussian-process surrogate models and a biophysical fitness formulation. 

Data:
- Download data_bloom and data_desai folders to get ESM embeddings and Kd values.
- Download script_results folder to get the variants selected by the model.
- Put these folders in the root directory.




![Schematic Overview](schematic.png)


## Repository layout

| Path                        | Purpose                                                                                                                                                                                                |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **src/active_learner.py**   | Core `ActiveLearner` class: keeps global–index bookkeeping, implements acquisition strategies (`random`, `greedy`, `UCB`), logs metrics, and exposes helpers to retrieve training-set histories. |
| **src/gaussian_process.py** | Lightweight GPyTorch wrapper that builds a squared-exponential ARD kernel, trains hyper-parameters, and returns mean / variance predictions for embedding vectors.                                     |
| **src/bio_model.py**        | Biophysical fitness layer. Provides `default_fitness`, its gradient, and `bio_model_var` for variance propagation from GP output to fitness space.                                                     |
| **src/utils.py**            | Utility collection: data loading/clean-up.                          |


---

## How to cite

If you use this toolkit in published work, please cite:

```bibtex
@article{Huot2025,
  author       = {Huot, Marian and Wang, Dianzhuo and Liu, Jiacheng and Shakhnovich, Eugene},
  title        = {Few-Shot Viral Variant Detection via Bayesian Active Learning and Biophysics},
  journal      = {bioRxiv},
  year         = {2025},
  elocation-id = {2025.03.12.642881},
  doi          = {10.1101/2025.03.12.642881},
  url          = {https://www.biorxiv.org/content/early/2025/03/13/2025.03.12.642881}
}
```
