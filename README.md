# Latent Data Assimilation for Global Atmosphere (1.41° resolution)

This repository contains the implementation of latent data assimilation (LDA) methods for the global atmosphere at a horizontal resolution of 1.41 degrees.

## Contents

- ✅ Autoencoder (AE) architecture
  📄 [`AutoEncoder.py`](./networks/AutoEncoder.py)
- ✅ AI forecast model to realize 4D DA method
  📄 [`forecast_net.py`](./networks/forecast_net.py)
- ✅ Utils for LDA experiments
  📄 [`exp_utils.py`](./LDA_Methods/exp_utils.py)
- ✅ A unified code for Latent 3D-Var (L3DVar) and Latent 4D-Var (L4DVar).
  📄 [`Latent_Var.py`](./LDA_Methods/Latent_Var.py)
- ✅ An Observing System Simulation Experiment (OSSE) example to apply LDA.
  📄 Example notebook: [`LDA_OSSEs.ipynb`](./DA_exps/LDA_OSSEs.ipynb)

The checkpoints for models are provided in https://zenodo.org/records/17210930.
