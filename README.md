# Evolutionary-based AutoML for Small Molecule Pharmacokinetic Prediction

This repository contains code and data for the paper "Evolutionary-based Automated Machine Learning for Small Molecule Pharmacokinetic Prediction", which has been accepted for publication and presentation at the [ 14th Workshop on Evolutionary Computation for the Automated Design of Algorithms (ECADA)](https://bonsai.auburn.edu/ecada/GECCO2024/).

For this paper, a grammar-based genetic programming (GGP) is used for searching and optimising machine learning (ML) pipelines in the context of small molecule pharmacokinetic (PK) prediction. Small Molecule Representation, Feature Scaling, Feature Selection, and ML Modelling are taking into account to compose the predictive ML-driven pipelines for PK.

## How to Install?

Our method uses Anaconda to install the requirements. It is worth noting we are relying on [alogos](https://github.com/robert-haas/alogos) for the basics on GGP.

`conda env create -f requirements.yaml`


## How to use the installed conda environment?

`conda activate automl4pk`

## How to use the AutoML method considering the Python code availabe?

After activating automl4pk environment, run:

`python automl4pk.py training_file.csv testing_file.csv seed_number num_cores output_dir`

E.g., using "datasets/01_caco2_train.csv" as the training set, "datasets/01_caco2_blindtest.csv" as the blind test, 1 as the seed number, 10 as the number of cores, and "." as the output directory, we would have:

`python automl4pk.py datasets/01_caco2_train.csv datasets/01_caco2_blindtest.csv 1 10 .`
