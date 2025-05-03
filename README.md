# WiDS 2025 Datathon - Age Prediction from Brain Connectomes

## 📖 Overview
This repository contains the code for participation in the WiDS Datathon 2025 University Edition challenge on Kaggle. The goal of this project is to build machine learning models that predict an individual’s age based on brain functional connectome matrices. The project utilizes data preprocessing, exploratory data analysis (EDA), traditional machine learning models, and deep learning approaches (including Graph Neural Networks).

## 📂 Project Structure

* **`Datapreprocessing.py`**:
    * Contains the `DataProcessor` class for loading correlation matrices from `.tsv` files and metadata from `.csv` files.
    * Includes methods to extract upper triangular elements from correlation matrices and process data into long or wide formats.
    * Provides functionality to merge correlation data with participant metadata.
    * Contains the `ExploratoryDataAnalysis` class for generating descriptive statistics, analyzing missing values, visualizing distributions, and exploring relationships between variables and factor scores.
    * Contains the `TrainModel` class which handles training and evaluation for Lasso, XGBoost (with hyperparameter tuning options), and ElasticNet models. It also includes functionality for SHAP value interpretation for XGBoost models.
* **`preprocessing_eda.ipynb`**:
    * Uses the `DataProcessor` and `ExploratoryDataAnalysis` classes from `Datapreprocessing.py`.
    * Loads training and test data from specified folders (`train_tsv1`, `test_tsv1`, `metadata/`).
    * Performs EDA including basic statistics, missing value analysis and visualization, correlation value distribution plots, participant-level analysis (mean/std correlation), demographic correlation heatmaps, and factor score analysis.
    * Includes code (commented out) to save the processed dataframes.
* **`modeling.ipynb`**:
    * Reads preprocessed data (e.g., `train_data_wide.csv`) and fills missing values.
    * Applies feature selection using `SelectKBest` with `f_regression`.
    * Applies dimensionality reduction using `PCA`.
    * Trains and evaluates a basic `LinearRegression` model.
    * Utilizes the `TrainModel` class from `Datapreprocessing.py` to train and evaluate Lasso, XGBoost (with options for hyperparameter tuning like `HalvingRandomSearchCV` or `GridSearchCV`), and Elasticnet models.
    * Includes model evaluation using cross-validation and checks regression assumptions (Residuals vs Fitted, QQ-plot).
* **`DeepLearning.py`**:
    * Defines PyTorch Graph Neural Network models: `AgeGCN` (Graph Convolutional Network), `AgeGAT` (Graph Attention Network), `AgeSAGE` (GraphSAGE).
    * Defines a function `make_graph` to convert correlation vectors (upper triangular) into PyTorch Geometric `Data` objects. This includes Fisher-transforming and standardizing correlations, thresholding the reconstructed matrix to create edges, and computing node features (strength, degree, pos/neg strength).
    * Defines MLP models (`AgeMLP`, `AgeMLP2` with BatchNorm and Dropout) for age prediction, likely intended for use with graph embeddings or other features.
    * Defines a Conditional Convolutional Autoencoder (`CorrCAE`) potentially for feature extraction or representation learning from the correlation matrices.
    * Includes an `objective` function for hyperparameter tuning using Optuna.
    * Contains code (likely for experimentation/training) to train the CAE model.
* **`README.md`**: This file.
* **`df_PCA.csv`**: Data file containing features after applying PCA.
* **`679 FP EDA.Rmd`**:
    * Performs data cleaning by removing rows with missing values in training and test datasets, ensuring complete cases for analysis.
    * Conducts exploratory data analysis (EDA) including participant counts before/after cleaning and visual comparisons using bar plots.
    * Analyzes demographic distributions (age, sex, race) with histograms, boxplots, and combined dataset visualizations.
    * Explores functional connectome data through feature-age correlations, UMAP dimensionality reduction, and average connectivity heatmaps stratified by sex.
* **`679 FP Modeling.Rmd`**:
    * Handles missing data via median imputation (BMI) and factor-level imputation (demographics), then engineers features using linear PCA and Nyström-PCA for dimensionality reduction.
    * Trains and evaluates multiple models (Linear Regression, Elastic Net, Lasso, Ridge, Random Forest, XGBoost) with validation splits, reporting RMSE and R² metrics.
    * Interprets key features using coefficient analysis (mapped to original connectome pairs) and SHAP values for model explainability.
    * Generates final age predictions on test data using an optimized Lasso model with linear PCA and exports results for deployment.


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

1.  **Data Setup**:
    * Place the training `.tsv` files in a folder named `train_tsv1`.
    * Place the test `.tsv` files in a folder named `test_tsv1`.
    * Place `training_metadata.csv` and `test_metadata.csv` in a folder named `metadata`.
2.  **Preprocessing and EDA**:
    * Run the `preprocessing_eda.ipynb` notebook. This will load the data, perform exploratory analysis, and can save the processed data (ensure the `to_csv` lines are uncommented if needed, e.g., to create `train_data_wide.csv`).
    * Run `679 FP EDA.Rmd` (R Markdown):
        * Performs complete-case filtering and generates `train_wide_complete.csv`/`test_wide_complete.csv`.
        * Creates exploratory visualizations (age distributions, UMAP plots, heatmaps) and saves them as PDFs.
3.  **Modeling (Traditional ML)**:
    * Run the `modeling.ipynb` notebook. This will:
        * Load the processed wide-format data.
        * Perform feature selection/dimensionality reduction (e.g., PCA).
        * Train and evaluate Linear Regression, Lasso, XGBoost, and ElasticNet models using the `TrainModel` class.
4.  **Modeling (Deep Learning)**:
    * Adapt and run code from `DeepLearning.py`. This might involve:
        * Using `make_graph` to create graph datasets from correlation vectors.
        * Training one of the GNN models (`AgeGCN`, `AgeGAT`, `AgeSAGE`).
        * Alternatively, training the CAE (`CorrCAE`) for feature extraction, possibly followed by an MLP (`AgeMLP`, `AgeMLP2`).
        * Using the `objective` function with Optuna for hyperparameter optimization if desired.
     * Run `679 FP Modeling.Rmd` (R Markdown):
        * Handles missing data (imputation for BMI/demographics).
        * Performs feature engineering with linear PCA and Nyström-PCA for dimensionality reduction.
        * Trains and validates models (Linear Regression, Elastic Net, Lasso, Ridge, Random Forest, XGBoost) with validation splits.
        * Outputs performance metrics (RMSE/R²) and feature importance plots (SHAP, VIP).
        * Generates final test predictions (`test_age_predictions.csv`) using the best model (Lasso + linear PCA).

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
