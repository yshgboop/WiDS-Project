# WiDS 2025 Datathon - Age Prediction from Brain Connectomes

## 📖 Overview
This repository contains the code for participation in the WiDS Datathon 2025 University Edition challenge on Kaggle. The goal of this project is to build machine learning models that predict an individual’s age based on brain functional connectome matrices. The project utilizes data preprocessing, exploratory data analysis (EDA), traditional machine learning models, and deep learning approaches (including Graph Neural Networks).

## 📂 Project Structure

* **`Datapreprocessing.py`**: Classes for data loading, processing, EDA, and training traditional ML models.
* **`preprocessing_eda.ipynb`**: Python notebook for initial data loading, EDA, and visualization.
* **`modeling.ipynb`**: Python notebook for feature selection, dimensionality reduction (PCA), and training/evaluating traditional ML models.
* **`DeepLearning.py`**: Python script defining PyTorch GNN, MLP, and CAE models, plus graph creation logic.
* **`679 FP EDA.Rmd`**: R Markdown for data cleaning, EDA (distributions, UMAP, heatmaps), and generating visualizations.
* **`679 FP Modeling.Rmd`**: R Markdown for imputation, feature engineering (PCA, Nyström-PCA), training/evaluating multiple ML models, feature interpretation (SHAP), and generating final predictions.


## ⚙️ Dependencies
* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scipy
* statsmodels
* scikit-learn
* category_encoders
* xgboost
* torch
* torch_geometric (for GNNs)
* optuna (for hyperparameter tuning)
* shap (for XGBoost interpretation)
* glob

## ▶️ How to Run

1. Set up data folders (`train_tsv1`, `test_tsv1`, `metadata`).
2. Run the preprocessing/EDA scripts (preprocessing_eda.ipynb, 679 FP EDA.Rmd).
3. Run the modeling scripts (modeling.ipynb, 679 FP Modeling.Rmd, DeepLearning.py).


## 📊 Models Implemented

**Traditional ML (R/Python)**

* Linear Models:
   * Linear Regression, Lasso (L1), Ridge (L2), ElasticNet
* Tree-Based Models:
   * Random Forest, XGBoost (with hyperparameter tuning)
* Feature Engineering:
   * Linear PCA, Nyström-PCA (approximate non-linear PCA)
* Interpretability:
   * SHAP values, coefficient mapping to original connectome features

**Deep Learning (Python)**

* Graph Neural Networks:
   * GCN, GAT, GraphSAGE
* Autoencoders:
   * Conditional Convolutional Autoencoder (CAE)
* MLPs:
   * AgeMLP, AgeMLP2
