import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import ElasticNetCV


class DataProcessor:
    def __init__(self, train_folder=None, test_folder=None, train_metadata_path=None, test_metadata_path=None):
        """
        Initialize the CorrelationProcessor with folder paths.
        
        Parameters:
        -----------
        train_folder : str
            Path to the folder containing training .tsv files
        test_folder : str
            Path to the folder containing test .tsv files
        train_metadata_path : str
            Path to the training metadata CSV file
        test_metadata_path : str
            Path to the test metadata CSV file
        """
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.train_metadata_path = train_metadata_path
        self.test_metadata_path = test_metadata_path
    
    def extract_upper_triangular(self, correlation_matrix):
        """
        Extract upper triangular elements (excluding diagonal) from correlation matrix.
        
        Parameters:
        -----------
        correlation_matrix : numpy.ndarray
            The correlation matrix to process
            
        Returns:
        --------
        numpy.ndarray
            1D array containing upper triangular elements
        """
        # Get indices of upper triangle (excluding diagonal)
        indices = np.triu_indices_from(correlation_matrix, k=1)
        # Extract values
        upper_triangular_values = correlation_matrix[indices]
        return upper_triangular_values
    
    def process_folder_long_format(self, folder_path):
        """
        Process all .tsv files in a folder and return a dataframe of correlation vectors.
        
        Parameters:
        -----------
        folder_path : str
            Path to the folder containing .tsv files
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with patient IDs and correlation features
        """
        all_files = glob(os.path.join(folder_path, "*.tsv"))
        all_rows = []
        
        for file_path in all_files:
            # Extract filename without path
            filename = os.path.basename(file_path)
            
            # Extract patient ID from the filename using the 'sub-' prefix pattern
            # This regex looks for 'sub-' followed by any characters until the next underscore
            import re
            match = re.search(r'sub-([^_]+)', filename)
            
            if match:
                patient_id = match.group(1)  # Extract the captured group (NDARAA306NT2)
            
            # Read the correlation matrix from .tsv file
            corr_matrix = pd.read_csv(file_path, sep='\t', header=None, index_col=None)
            # print(corr_matrix.shape)
            
            # Extract upper triangular elements
            corr_vector = self.extract_upper_triangular(corr_matrix.values)
            
            for i, val in enumerate(corr_vector):
                # Each row contains participant ID and a single correlation value
                row = {
                    "participant_id": patient_id,
                    "correlation_id": f"corr_{i+1}",
                    "correlation_value": val
                }
                all_rows.append(row)
        
        # Create dataframe from all processed files
        return pd.DataFrame(all_rows)
    
    def process_folder_wide_format(self, folder_path):
        """
        Process all .tsv files in a folder and return a dataframe of correlation vectors.
        
        Parameters:
        -----------
        folder_path : str
            Path to the folder containing .tsv files
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with patient IDs and correlation features
        """
        all_files = glob(os.path.join(folder_path, "*.tsv"))
        all_rows = []
        
        for file_path in all_files:
            # Extract filename without path
            filename = os.path.basename(file_path)
            
            # Extract patient ID from the filename using the 'sub-' prefix pattern
            # This regex looks for 'sub-' followed by any characters until the next underscore
            import re
            match = re.search(r'sub-([^_]+)', filename)
            
            if match:
                patient_id = match.group(1)  # Extract the captured group (NDARAA306NT2)
            
            # Read the correlation matrix from .tsv file
            corr_matrix = pd.read_csv(file_path, sep='\t', header=None, index_col=None)
            # print(corr_matrix.shape)
            
            # Extract upper triangular elements
            corr_vector = self.extract_upper_triangular(corr_matrix.values)
            
            for i, val in enumerate(corr_vector):
                # Each row contains participant ID and a single correlation value
                row = {
                    "participant_id": patient_id,
                    "correlation_id": f"corr_{i+1}",
                    "correlation_value": val
                }
                all_rows.append(row)
        
        # Create dataframe from all processed files
        return pd.DataFrame(all_rows)
    
    def process_folder_wide_format(self, folder_path):
        """
        Process all .tsv files in a folder and return a dataframe in wide format,
        where each participant has one row with all correlation values as columns.
        
        Parameters:
        -----------
        folder_path : str
            Path to the folder containing .tsv files
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with participant IDs (rows) and correlation features (columns)
        """
        all_files = glob(os.path.join(folder_path, "*.tsv"))
        result_data = []
        
        for file_path in all_files:
            # Extract filename without path
            filename = os.path.basename(file_path)
            
            # Extract patient ID from the filename using the 'sub-' prefix pattern
            import re
            match = re.search(r'sub-([^_]+)', filename)
            
            if match:
                patient_id = match.group(1)  # Extract the captured group (NDARAA306NT2)
            
            # Read the correlation matrix from .tsv file
            corr_matrix = pd.read_csv(file_path, sep='\t', header=None, index_col=None)
            
            # Extract upper triangular elements
            corr_vector = self.extract_upper_triangular(corr_matrix.values)
            
            # Create a dictionary with participant ID and all correlation values
            patient_data = {"participant_id": patient_id}
            
            # Add correlation features as columns (wide format)
            for i, val in enumerate(corr_vector):
                patient_data[f"corr_{i+1}"] = val
                
            result_data.append(patient_data)
        
        # Create dataframe from all processed files
        return pd.DataFrame(result_data)
    
    
    def prepare_datasets(self, wide_format=True):
        """
        Process training and test folders and merge with metadata.

        Parameters:
        -----------
        wide_format : bool
            If True, process data in wide format; otherwise, in long format.
        
        Returns:
        --------
        tuple
            (train_data, test_data) - DataFrames with correlation features and metadata
        """
        if not all([self.train_folder, self.test_folder, self.train_metadata_path, self.test_metadata_path]):
            raise ValueError("Folder paths and metadata paths must be set before preparing datasets")
        
        # Process training and test folders
        train_df_long = self.process_folder_long_format(self.train_folder)
        test_df_long = self.process_folder_long_format(self.test_folder)
        train_df_wide = self.process_folder_wide_format(self.train_folder)
        test_df_wide = self.process_folder_wide_format(self.test_folder)
        
        # Load metadata
        train_metadata = pd.read_csv(self.train_metadata_path)
        test_metadata = pd.read_csv(self.test_metadata_path)
        
        # Merge with metadata
        self.train_data_long = pd.merge(train_df_long, train_metadata, on="participant_id")
        self.test_data_long = pd.merge(test_df_long, test_metadata, on="participant_id")
        self.train_data_wide = pd.merge(train_df_wide, train_metadata, on="participant_id")
        self.test_data_wide = pd.merge(test_df_wide, test_metadata, on="participant_id")
        
        if wide_format:
            self.train_data = self.train_data_wide
            self.test_data = self.test_data_wide
        else:
            self.train_data = self.train_data_long
            self.test_data = self.test_data_long
            
        return self.train_data, self.test_data
    
    
    def get_feature_names(self):
        """
        Get the names of the correlation features.
        
        Returns:
        --------
        list
            List of feature column names
        """
        if self.train_data is None:
            raise ValueError("Datasets not prepared yet. Call prepare_datasets() first.")
        
        corr_train = [col for col in self.train_data.columns if col.startswith('corr_')]
        corr_test = [col for col in self.test_data.columns if col.startswith('corr_')]
        
        return corr_train, corr_test



class ExploratoryDataAnalysis:
    """
    A class to perform exploratory data analysis on correlation datasets in long format.
    
    In the long format:
    - Each participant has multiple rows (one per correlation coefficient)
    - Demographic information is duplicated across these rows
    - Structure: participant_id, correlation_id, correlation_value, demographic variables
    """
    
    def __init__(self, file_path=None, dataframe=None):
        """
        Initialize the EDA class with either a file path or a pandas DataFrame.
        
        Parameters:
        -----------
        file_path : str, optional
            Path to the CSV file to analyze
        dataframe : pandas.DataFrame, optional
            Pandas DataFrame to analyze
        """
        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("viridis")
        plt.rcParams.update({'font.size': 12})
        
        # Load data
        if dataframe is not None:
            self.df = dataframe
        elif file_path is not None:
            self.df = pd.read_csv(file_path)
        else:
            raise ValueError("Either file_path or dataframe must be provided")
        
        # Check if data is in long format (expecting columns: participant_id, correlation_id, correlation_value)
        required_cols = ['participant_id', 'correlation_id', 'correlation_value']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError("Data does not appear to be in long format. Expected columns: participant_id, correlation_id, correlation_value")
        
        # Identify key columns
        self.id_column = 'participant_id'
        self.corr_id_column = 'correlation_id'
        self.corr_value_column = 'correlation_value'
        
        # Identify column types (excluding correlation-specific columns)
        self.demographic_cols = [col for col in self.df.columns if col not in required_cols]
        
        # Further categorize demographic columns
        self.numerical_demo_cols = self.df[self.demographic_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_demo_cols = self.df[self.demographic_cols].select_dtypes(include=['object']).columns.tolist()
        
        # Identify factor scores (special numerical columns ending with _fs)
        self.factor_scores = [col for col in self.numerical_demo_cols if col.endswith('_fs')]
        
        # Get unique participants
        self.unique_participants = self.df[self.id_column].unique()
        print(f"Dataset contains {len(self.unique_participants)} unique participants with {len(self.df)} total observations")

    def inspect_data(self):
        """
        Basic inspection of the dataset structure and content.
        """
        print("=" * 50)
        print("Data Overview:")
        print("=" * 50)
        
        # Print basic information
        print(f"Dataset shape: {self.df.shape}")
        print(f"Unique participants: {len(self.unique_participants)}")
        print(f"Average rows per participant: {len(self.df) / len(self.unique_participants):.2f}")
        
        # Print column information
        print("\nColumn types:")
        print(f"Demographic columns: {len(self.demographic_cols)}")
        print(f"- Numerical: {len(self.numerical_demo_cols)}")
        print(f"- Categorical: {len(self.categorical_demo_cols)}")
        
        # Print the first few rows
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        # Print unique correlation IDs (first few)
        unique_corr_ids = self.df[self.corr_id_column].unique()
        print(f"\nTotal unique correlation IDs: {len(unique_corr_ids)}")
        print(f"First 5 correlation IDs: {unique_corr_ids[:5]}")
        
        return self
    
    def basic_stats(self):
        """
        Calculate and display basic descriptive statistics.
        """
        print("=" * 50)
        print("Basic Statistics:")
        print("=" * 50)
        
        # Stats for correlation values
        print("\nCorrelation Values Statistics:")
        print(self.df[self.corr_value_column].describe())
        
        # Create a participant-level dataset for demographic statistics
        # (take the first occurrence of each participant to avoid duplication)
        participant_df = self.df.drop_duplicates(subset=[self.id_column])
        
        # Numerical demographics statistics
        print("\nDemographic Variables Statistics (Participant Level):")
        if self.numerical_demo_cols:
            print(participant_df[self.numerical_demo_cols].describe().T)
        else:
            print("No numerical demographic variables found.")
        
        # Categorical demographics statistics
        print("\nCategorical Variables Value Counts (Participant Level):")
        for col in self.categorical_demo_cols:
            print(f"\n{col} Value Counts:")
            print(participant_df[col].value_counts())
            print(f"{col} Unique Values: {participant_df[col].nunique()}")
        
        return self
    
    def missing_values_analysis(self):
        """
        Analyze missing values in the dataset.
        """
        print("=" * 50)
        print("Missing Values Analysis:")
        print("=" * 50)
        
        # Check for missing values in correlation data
        corr_missing = self.df[self.corr_value_column].isnull().sum()
        print(f"Missing correlation values: {corr_missing} ({corr_missing/len(self.df):.2%})")
        
        # Create a participant-level dataset for demographic missing values
        participant_df = self.df.drop_duplicates(subset=[self.id_column])
        
        # Count missing values per demographic column
        missing_values = participant_df[self.demographic_cols].isnull().sum()
        missing_values_percent = (participant_df[self.demographic_cols].isnull().sum() / len(participant_df)) * 100
        
        # Create a dataframe with missing values information
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage (%)': missing_values_percent
        })
        
        missing_data = missing_df[missing_df['Missing Values'] > 0]
        
        if len(missing_data) > 0:
            print("\nMissing Values in Demographic Data (Participant Level):")
            print(missing_data)
            
            # Plot missing values for demographics
            plt.figure(figsize=(12, 6))
            ax = sns.heatmap(participant_df[self.demographic_cols].isnull(), 
                         cbar=False, cmap='viridis', yticklabels=False)
            plt.title('Missing Values in Demographic Data')
            plt.tight_layout()
            plt.show()
        else:
            print("\nNo missing values found in demographic data.")
        
        return self
    
    def correlation_distribution(self):
        """
        Analyze the distribution of correlation values.
        """
        print("=" * 50)
        print("Correlation Values Distribution:")
        print("=" * 50)
        
        # Overall distribution of correlation values
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df[self.corr_value_column].dropna(), kde=True)
        plt.title('Distribution of All Correlation Values')
        plt.xlabel('Correlation Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
        # Distribution statistics
        print("\nDistribution Statistics for Correlation Values:")
        print(self.df[self.corr_value_column].describe())
        
        # Check for extreme values
        q1 = self.df[self.corr_value_column].quantile(0.25)
        q3 = self.df[self.corr_value_column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = self.df[(self.df[self.corr_value_column] < lower_bound) | 
                           (self.df[self.corr_value_column] > upper_bound)]
        
        print(f"\nPotential outliers: {len(outliers)} values outside of range [{lower_bound:.3f}, {upper_bound:.3f}]")
        
        # Create a boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=self.corr_value_column, data=self.df)
        plt.title('Boxplot of Correlation Values')
        plt.tight_layout()
        plt.show()
        
        return self
    
    def participant_level_analysis(self):
        """
        Analyze correlation patterns at the participant level.
        """
        print("=" * 50)
        print("Participant-Level Analysis:")
        print("=" * 50)
        
        # Calculate statistics for each participant
        participant_stats = self.df.groupby(self.id_column)[self.corr_value_column].agg(['mean', 'std', 'min', 'max', 'count'])
        
        # Visualize distribution of participant means
        plt.figure(figsize=(12, 6))
        sns.histplot(participant_stats['mean'], kde=True)
        plt.title('Distribution of Mean Correlation Values Across Participants')
        plt.xlabel('Mean Correlation Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
        # Visualize distribution of participant standard deviations
        plt.figure(figsize=(12, 6))
        sns.histplot(participant_stats['std'], kde=True)
        plt.title('Distribution of Standard Deviations in Correlation Values Across Participants')
        plt.xlabel('Standard Deviation of Correlation Values')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
        return self
    
    
    def demographic_correlation_heatmap(self, demo_vars=None):
        """
        Create a heatmap showing how demographic variables correlate with each other.
        
        Parameters:
        -----------
        demo_vars : list, optional
            List of demographic variables to include in the heatmap
        """
        # Create a participant-level dataset
        participant_df = self.df.drop_duplicates(subset=[self.id_column])
        
        # Select variables for correlation
        if demo_vars is None:
            # Use all numerical demographic variables
            demo_vars = self.numerical_demo_cols
        else:
            # Filter to ensure all variables exist and are numerical
            demo_vars = [var for var in demo_vars if var in self.numerical_demo_cols]
        
        if len(demo_vars) < 2:
            print("Not enough numerical demographic variables for correlation analysis.")
            return self
        
        print("=" * 50)
        print("Demographic Variables Correlation Heatmap:")
        print("=" * 50)
        
        # Calculate correlation matrix
        corr_matrix = participant_df[demo_vars].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=.5,
            annot=True,
            fmt=".2f"
        )
        
        plt.title('Correlation Matrix of Demographic Variables')
        plt.tight_layout()
        plt.show()
        
        return self
    
    def factor_scores_analysis(self):
        """
        Analyze how factor scores relate to correlation patterns.
        """
        if not self.factor_scores:
            print("No factor scores found in the dataset.")
            return self
        
        print("=" * 50)
        print("Factor Scores and Correlation Analysis:")
        print("=" * 50)
        
        # Analyze each factor score
        for factor in self.factor_scores:
            print(f"\nAnalysis for {factor}:")
            
            # Calculate mean correlation value for each participant
            participant_means = self.df.groupby(self.id_column)[self.corr_value_column].mean().reset_index()
            
            # Get factor scores (from the first occurrence of each participant)
            factor_data = self.df.drop_duplicates(subset=[self.id_column])[[self.id_column, factor]]
            
            # Merge data
            merged_data = pd.merge(participant_means, factor_data, on=self.id_column)
            
            # Calculate correlation between factor score and mean correlation value
            corr, p_val = stats.pearsonr(merged_data[factor], merged_data[self.corr_value_column])
            
            print(f"Correlation between {factor} and mean correlation value: {corr:.4f}, p-value: {p_val:.4f}")
            
            # Create scatter plot
            plt.figure(figsize=(10, 6))
            sns.regplot(x=factor, y=self.corr_value_column, data=merged_data, scatter_kws={'alpha':0.5})
            plt.title(f'{factor} vs Mean Correlation Value (r = {corr:.4f}, p = {p_val:.4f})')
            plt.xlabel(factor)
            plt.ylabel('Mean Correlation Value')
            plt.tight_layout()
            plt.show()
            
            # Divide participants into high/low factor score groups
            median_factor = merged_data[factor].median()
            merged_data['factor_group'] = np.where(merged_data[factor] > median_factor, 'High', 'Low')
            
            # Compare correlation values between high/low groups
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='factor_group', y=self.corr_value_column, data=merged_data)
            plt.title(f'Correlation Values by {factor} Group (Split at Median)')
            plt.xlabel(f'{factor} Group')
            plt.ylabel('Mean Correlation Value')
            plt.tight_layout()
            plt.show()
            
            # T-test between high/low groups
            high_group = merged_data[merged_data['factor_group'] == 'High'][self.corr_value_column]
            low_group = merged_data[merged_data['factor_group'] == 'Low'][self.corr_value_column]
            
            t_stat, p_val = stats.ttest_ind(high_group, low_group, equal_var=False)
            
            print(f"T-test between high and low {factor} groups: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
            
            if p_val < 0.05:
                print(f"The difference in correlation values between high and low {factor} groups is statistically significant (p < 0.05).")
            else:
                print(f"The difference in correlation values between high and low {factor} groups is not statistically significant (p >= 0.05).")
        
        return self


from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
import category_encoders as ce

def evaluate_regression_model(X, y, model_type=LinearRegression, 
                              n_splits=5, random_state=724, 
                              return_train_score=True):
    """
    Evaluate a regression model with cross-validation, automatically handling
    categorical features with target encoding if needed.
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature dataframe
    y : pandas Series or array-like
        Target variable
    model_type : scikit-learn estimator, default=LinearRegression
        The regression model to use
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random state for reproducibility
    return_train_score : bool, default=True
        Whether to return training scores
        
    Returns:
    --------
    dict : Cross-validation results
    """
    # Detect if there are categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    has_categorical = len(categorical_cols) > 0
    
    # Create model pipeline based on data types
    if has_categorical:
        # With target encoding for categorical data
        model = Pipeline([
            ('encoder', ce.TargetEncoder()),
            ('regressor', model_type())
        ])
        print("Using target encoding pipeline")
    else:
        # Without encoder for numeric-only data
        model = model_type()
        print("Using direct model (no encoding needed)")
    # 2) fit on the entire X, y
    model.fit(X, y)

    # 3) compute residuals
    y_pred    = model.predict(X)
    residuals = y - y_pred

    # 4) now run the 4 assumption checks:
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import het_breuschpagan
    import matplotlib.pyplot as plt

    # (a) Residuals vs fitted
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='grey', linestyle='--')
    plt.xlabel('Fitted values'); plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted')
    plt.show()

    # (b) QQ‑plot
    sm.qqplot(residuals, line='45', fit=True)
    plt.title('Q-Q Plot')
    plt.show()
    
    # Set up cross-validation
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
    
    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                return_train_score=return_train_score)
    
    # Print results
    print("\nCross-Validation Results:")
    
    if return_train_score:
        print(f"Train R² Scores: {cv_results['train_r2']}")
        print(f"Mean Train R² Score: {np.mean(cv_results['train_r2']):.4f}")
    
    print(f"Test R² Scores: {cv_results['test_r2']}")
    print(f"Mean Test R² Score: {np.mean(cv_results['test_r2']):.4f}")
    print(f"Test MSE Scores: {-cv_results['test_mse']}") # Negate to get actual MSE
    print(f"Mean Test MSE: {-np.mean(cv_results['test_mse']):.4f}")

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, train_test_split, GridSearchCV
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

class TrainModel:
    """
    A class that handles training, evaluation and running of Lasso regression models.
    """
    
    def __init__(self, random_state=724):
        """Initialize the LassoModel with a random state for reproducibility."""
        self.random_state = random_state
        self.best_model = None
        self.results = {
            'best_params': [],
            'r2_train_scores': [],
            'r2_test_scores': [],
            'mse_test_scores': []
        }
    
    def train_lasso(self, X_train, y_train, cv=None, alphas=None, max_iter=10000, tol=0.0001):
        """
        Train a Lasso model with cross-validation.
        
        Parameters:
        -----------
        X_train : DataFrame - Training features
        y_train : Series - Training target
        cv : int or cross-validation generator, default=None
        alphas : array-like, default=None - List of alphas to try
        max_iter : int, default=10000 - Maximum number of iterations
        tol : float, default=0.0001 - Tolerance for optimization
        
        Returns:
        --------
        self - For method chaining
        """
        if alphas is None:
            alphas = np.logspace(-4, 1, 50)
            
        # Train LassoCV model
        self.best_lasso_model = LassoCV(
            alphas=alphas,
            cv=cv,
            max_iter=max_iter,
            tol=tol,
            random_state=self.random_state,
            precompute=False
        ).fit(X_train, y_train)
        
        return self
    
    # def train_xgboost(
    #     self,
    #     X_train,
    #     y_train,
    #     param_grid=None,
    #     param_grid_choice='small',      # default to the smaller grid
    #     cv_folds=5,
    #     random_state=None
    # ):
        
    #     rs = random_state or self.random_state

    #     # Tighter predefined grid:
    #     predefined_grids = {
    #         'small': {
    #             'max_depth':       [3, 4, 5],
    #             'learning_rate':   [0.01, 0.05, 0.1],
    #             'subsample':       [0.5, 0.7, 0.8],
    #             'colsample_bytree':[0.3, 0.5, 0.7],
    #             'gamma':           [0.1, 0.3, 0.5],
    #             'reg_alpha':       [0, 0.1, 1],
    #             'reg_lambda':      [1, 5, 10],
    #             'booster':         ['gbtree']
    #         }
    #     }

    #     # choose grid
    #     if param_grid is not None:
    #         grid = param_grid
    #     else:
    #         grid = predefined_grids.get(param_grid_choice, predefined_grids['small'])

    #     # CV splitter
    #     kf = KFold(n_splits=cv_folds, shuffle=True, random_state=rs)

    #     base = XGBRegressor(
    #         objective='reg:squarederror',
    #         random_state=rs,
    #         n_jobs=-1,
    #         # push for early stopping on fewer rounds:
    #         early_stopping_rounds=20
    #     )

    #     search = HalvingRandomSearchCV(
    #         estimator=base,
    #         param_distributions=grid,
    #         factor=3,
    #         resource='n_estimators',
    #         max_resources=200,            # lower ceiling on trees
    #         cv=kf,
    #         scoring='r2',
    #         random_state=rs,
    #         verbose=1
    #     )

    #     X_main, X_val, y_main, y_val = train_test_split(
    #         X_train, y_train, test_size=0.1, random_state=rs
    #     )
    #     search.fit(
    #         X_main, y_main,
    #         eval_set=[(X_val, y_val)],
    #         verbose=False
    #     )

    #     best = search.best_estimator_
    #     best_params = search.best_params_.copy()
    #     best_params.setdefault('n_estimators', best.get_params()['n_estimators'])

    #     self.best_xgb = best
    #     self.best_xgb_params = best_params

    def train_xgboost(self, X_train, y_train, param_grid_choice='small', cv_folds=10, random_state=None):
        """
        Train an XGBoost model with hyperparameter tuning.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Training target
        param_grid : dict, default=None
            Custom parameter grid for hyperparameter tuning
        param_grid_choice : str, default=None
            Choice of predefined parameter grid ('small', 'large')
        cv_folds : int, default=5
            Number of cross-validation folds
        random_state : int, default=None
            Random state for the model
            
        Returns:
        --------
        tuple
            (best_estimator, best_params)
        """
        if random_state is None:
            random_state = self.random_state
            
        # Define predefined parameter grids
        predefined_grids = {
            'small': {
                'n_estimators': [100, 200, 500],
                'max_depth': [2, 3, 4],
                'learning_rate': [0.01, 0.03, 0.05],
                'subsample': [0.6, 0.8, 1.0]
            },
            'large': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [2, 3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [3, 5, 7],
                'reg_alpha': [0, 0.1, 1],  # L1 regularization
                'reg_lambda': [1, 1.5, 2]  # L2 regularization
            }
        }
        
        # Determine which parameter grid to use
        if param_grid_choice == 'small':
            # User-provided grid takes precedence
            selected_param_grid = predefined_grids['small']
        else:
            # Default to small grid
            selected_param_grid = predefined_grids['large']
            
        # Define cross-validation strategy
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Define base XGBoost model
        xgb_base = XGBRegressor(
            random_state=random_state,
            objective='reg:squarederror',
            early_stopping_rounds=10,
            eval_metric='rmse'
        )
        
        # GridSearchCV to find best parameters
        grid_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=selected_param_grid,
            cv=kf,
            scoring='r2',
            n_jobs=-1,
            verbose=0,
            return_train_score=True
        )
        
        # Create an evaluation set from training data for early stopping
        X_train_main, X_eval, y_train_main, y_eval = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state
        )
        
        # Fit with evaluation set
        grid_search.fit(
            X_train_main, y_train_main,
            eval_set=[(X_eval, y_eval)],
            verbose=False
        )
        
        # Get best parameters and model
        self.best_params = grid_search.best_params_
        self.best_xgb = grid_search.best_estimator_
        
        return self.best_xgb, self.best_params
    

    def train_elasticnet(
        self,
        X_train,
        y_train,
        cv=10,
        l1_ratio_list=None,
        alphas=None,
        max_iter=10000,
        tol=0.0001,
        n_jobs=-1
    ):
        """
        Train an ElasticNet model with cross-validation.

        Parameters:
        -----------
        X_train : array-like or DataFrame of shape (n_samples, n_features)
            Training features.
        y_train : array-like or Series of shape (n_samples,)
            Training target.
        cv : int or cross-validation generator, default=None
            Number of folds or CV splitter.
        l1_ratio_list : list of floats, default=None
            List of l1_ratio values to try (1.0 = Lasso, 0.0 = Ridge).
        alphas : array-like, default=None
            List of alpha values to try.
        max_iter : int, default=10000
            Maximum number of iterations for the solver.
        tol : float, default=1e-4
            Tolerance for the optimization.
        n_jobs : int, default=-1
            Number of CPUs to use for the cross-validation.

        Returns:
        --------
        self : object
            Returns self for method chaining.
        """
        # sensible defaults
        if l1_ratio_list is None:
            l1_ratio_list = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        if alphas is None:
            alphas = np.logspace(-4, 1, 50)

        self.best_elasticnet_model = ElasticNetCV(
            l1_ratio=l1_ratio_list,
            alphas=alphas,
            cv=cv,
            max_iter=max_iter,
            tol=tol,
            random_state=self.random_state,
            n_jobs=n_jobs,
            precompute=False
        ).fit(X_train, y_train)

        return self



    def evaluate(self, X_train, X_test, y_train, y_test, store_results=True, 
                 evallasso=False, evalxgb = True, evalelasticnet=False):
        """
        Evaluate model performance on train and test sets.
        
        Parameters:
        -----------
        X_train, X_test : DataFrames - Training and testing features
        y_train, y_test : Series - Training and testing targets
        store_results : bool, default=True - Whether to store results in instance
        
        Returns:
        --------
        dict - Metrics dictionary
        """
        if evalxgb:
            self.best_model = self.best_xgb
        elif evalelasticnet:
            self.best_model = self.best_elasticnet_model
        else:
            self.best_model = self.best_lasso_model
            
        # Predict on training and test sets
        best_pred_train = self.best_model.predict(X_train)
        best_pred_test = self.best_model.predict(X_test)
        
        # Calculate performance metrics
        r2_best_train = r2_score(y_train, best_pred_train)
        r2_best_test = r2_score(y_test, best_pred_test)
        mse_best_test = mean_squared_error(y_test, best_pred_test)
        
        # Get model parameters
        if evallasso:
            print(f"Best alpha value: {self.best_model.alpha_}")
        elif evalelasticnet:
            print(f"Best alpha value: {self.best_model.alpha_}")
            print(f"Best l1_ratio value: {self.best_model.l1_ratio_}")
        else:
            print(self.best_params)
        
        # Store results if requested
        if store_results:
            self.results['r2_train_scores'].append(r2_best_train)
            self.results['r2_test_scores'].append(r2_best_test)
            self.results['mse_test_scores'].append(mse_best_test)
        
        # Return metrics
        metrics = {
            'r2_train': r2_best_train,
            'r2_test': r2_best_test,
            'mse_test': mse_best_test
        }
        
        return metrics
    
    def run(self, X, y, test_size=0.2, cv=10, trainlasso=False, train_xgb = True, train_elastic = False):
        """
        Run the full training and evaluation process.
        
        Parameters:
        -----------
        X : DataFrame - Feature dataframe
        y : Series - Target variable
        test_size : float, default=0.2 - Proportion of test data
        cv : int, default=10 - Number of cross-validation folds
        
        Returns:
        --------
        tuple - (metrics, model)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Train model
        if trainlasso:
            self.train_lasso(X_train, y_train, cv=cv)
        if train_elastic:
            self.train_elasticnet(X_train, y_train)
        if train_xgb:
            self.train_xgboost(X_train, y_train)
        
        # Evaluate model
        if train_xgb:
            metrics = self.evaluate(X_train, X_test, y_train, y_test)
        if trainlasso:
            metrics = self.evaluate(X_train, X_test, y_train, y_test, evallasso=True, evalxgb=False)
        if train_elastic:
            metrics = self.evaluate(X_train, X_test, y_train, y_test, evalelasticnet=True, evalxgb=False)
        
        # Print results
        print("\n--- Model Evaluation Results ---")
        if trainlasso:
            print(f"Lasso - R² Train: {metrics['r2_train']:.4f}, "
                f"R² Test: {metrics['r2_test']:.4f}, "
                f"MSE Test: {metrics['mse_test']:.4f}")
        if train_xgb:
            print(f"XGBoost - R² Train: {metrics['r2_train']:.4f}, "
                f"R² Test: {metrics['r2_test']:.4f}, "
                f"MSE Test: {metrics['mse_test']:.4f}")
        if train_elastic:
            print(f"ElasticNet - R² Train: {metrics['r2_train']:.4f}, "
                f"R² Test: {metrics['r2_test']:.4f}, "
                f"MSE Test: {metrics['mse_test']:.4f}")
        
        return metrics, self.best_model


import shap

def interpret_xgboost_with_shap(best_model, X, feature_names=None, output_dir=None):
    """
    Interpret an XGBoost model using SHAP values.
    
    Parameters:
    -----------
    best_model : XGBoost model
        The best model already trained through cross-validation
    X : DataFrame or array
        Feature matrix used for generating SHAP values
    feature_names : list, optional
        List of feature names. If None and X is not a DataFrame, will use generic names
    output_dir : str, optional
        Directory to save plots. If None, plots will only be displayed
        
    Returns:
    --------
    shap_values : shap.Explanation
        SHAP values for each prediction
    """
    # Set feature names if not provided
    if feature_names is None:
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    explainer = shap.Explainer(best_model)
    shap_values = explainer(X)
    
    # Create SHAP plots
    print("Creating SHAP plots...")
    
    # Beeswarm summary plot (distribution of impacts)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Feature Impact Distribution")
    plt.tight_layout()
    plt.show()

import numpy as np
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin

class CPMTransformer(BaseEstimator, TransformerMixin):
    """
    For each fold, selects edges whose correlation with y is significant (p < p_thresh),
    splits them into pos/neg sets, then transforms X to two features:
      [sum of pos-edges, sum of neg-edges].
    """
    def __init__(self, p_thresh=0.01):
        self.p_thresh = p_thresh

    def fit(self, X, y):
        # X: (n_samples, n_edges), y: (n_samples,)
        rs = []
        ps = []
        for i in range(X.shape[1]):
            r, p = pearsonr(X[:, i], y)
            rs.append(r)
            ps.append(p)
        rs = np.array(rs)
        ps = np.array(ps)

        # store indices of edges to sum
        self.pos_idx_ = np.where((rs > 0) & (ps < self.p_thresh))[0]
        self.neg_idx_ = np.where((rs < 0) & (ps < self.p_thresh))[0]
        return self

    def transform(self, X):
        # collapse each subject to two network‐strength features
        pos_sum = X[:, self.pos_idx_].sum(axis=1)
        neg_sum = X[:, self.neg_idx_].sum(axis=1)
        return np.vstack([pos_sum, neg_sum]).T
