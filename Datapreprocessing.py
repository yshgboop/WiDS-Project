import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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
        self.train_data = None
        self.test_data = None
    
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
    
    def process_folder(self, folder_path):
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
    
    def prepare_datasets(self):
        """
        Process training and test folders and merge with metadata.
        
        Returns:
        --------
        tuple
            (train_data, test_data) - DataFrames with correlation features and metadata
        """
        if not all([self.train_folder, self.test_folder, self.train_metadata_path, self.test_metadata_path]):
            raise ValueError("Folder paths and metadata paths must be set before preparing datasets")
        
        # Process training and test folders
        train_df = self.process_folder(self.train_folder)
        test_df = self.process_folder(self.test_folder)
        
        # Load metadata
        train_metadata = pd.read_csv(self.train_metadata_path)
        test_metadata = pd.read_csv(self.test_metadata_path)
        
        # Merge with metadata
        self.train_data = pd.merge(train_df, train_metadata, on="participant_id")
        self.test_data = pd.merge(test_df, test_metadata, on="participant_id")
        
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
    
    def get_data(self):
        """
        Get the prepared training and test datasets.
        
        Returns:
        --------
        tuple
            (train_data, test_data) - Prepared DataFrames
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("Datasets not prepared yet. Call prepare_datasets() first.")
        
        return self.train_data, self.test_data



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