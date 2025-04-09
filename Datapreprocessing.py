import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    A class to perform comprehensive exploratory data analysis on a dataset.
    
    This class provides methods for various types of analyses including:
    - Basic statistics and data inspection
    - Missing values analysis
    - Distribution analysis
    - Correlation analysis
    - Group comparisons
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
        
        # Identify column types
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Identify special column types
        self.factor_scores = [col for col in self.numerical_cols if col.endswith('_fs')]
        self.id_column = 'participant_id' if 'participant_id' in self.df.columns else None

    
    def basic_stats(self):
        """
        Calculate and display basic descriptive statistics for numerical 
        and categorical variables.
        """
        # Numerical statistics
        print("=" * 50)
        print("Numerical Variables Statistics:")
        print("=" * 50)
        print(self.df[self.numerical_cols].describe().T)
        
        # Categorical statistics
        print("\n" + "=" * 50)
        print("Categorical Variables Value Counts:")
        print("=" * 50)
        for col in self.categorical_cols:
            if col != self.id_column:  # Skip ID column
                print(f"\n{col} Value Counts:")
                print(self.df[col].value_counts())
                print(f"{col} Unique Values: {self.df[col].nunique()}")
        
        return self
    
    def missing_values_analysis(self):
        """
        Analyze missing values in the dataset.
        """
        print("=" * 50)
        print("Missing Values Analysis:")
        print("=" * 50)
        
        # Count missing values per column
        missing_values = self.df.isnull().sum()
        missing_values_percent = (self.df.isnull().sum() / len(self.df)) * 100
        
        # Create a dataframe with missing values information
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage (%)': missing_values_percent
        })
        
        missing_data = missing_df[missing_df['Missing Values'] > 0]
        
        if len(missing_data) > 0:
            print(missing_data)
            
            # Plot missing values
            plt.figure(figsize=(12, 6))
            sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
            plt.title('Missing Values Heatmap')
            plt.tight_layout()
            plt.show()
        else:
            print("No missing values found in the dataset.")
        
        return self
    
    def distribution_analysis(self):
        """
        Analyze and visualize the distribution of numerical variables.
        """
        print("=" * 50)
        print("Distribution Analysis:")
        print("=" * 50)
        
        # Histograms for numerical variables
        num_plots = len(self.numerical_cols)
        fig_rows = (num_plots + 1) // 2  # Calculate number of rows needed
        
        plt.figure(figsize=(15, 5 * fig_rows))
        
        for i, col in enumerate(self.numerical_cols, 1):
            plt.subplot(fig_rows, 2, i)
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
        
        plt.show()
        
        # Boxplots for factor scores
        if self.factor_scores:
            plt.figure(figsize=(15, 10))
            
            # Melt the dataframe to get it in the right format for seaborn
            melted_df = self.df[self.factor_scores].melt()
            
            # Create the boxplot
            sns.boxplot(x='variable', y='value', data=melted_df)
            plt.title('Boxplots of Factor Scores')
            plt.xlabel('Factor Score')
            plt.ylabel('Value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # QQ plots for normality check
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            axes = axes.flatten()
            
            for i, col in enumerate(self.factor_scores):
                if i < len(axes):  # Ensure we don't try to access non-existent axes
                    stats.probplot(self.df[col].dropna(), plot=axes[i])
                    axes[i].set_title(f'Q-Q Plot of {col}')
            
            plt.tight_layout()
            plt.show()
        
        return self
    
    def categorical_visualization(self):
        """
        Visualize categorical variables using bar charts.
        """
        print("=" * 50)
        print("Categorical Variables Visualization:")
        print("=" * 50)
        
        # Select categorical columns with a reasonable number of unique values
        selected_cats = []
        for col in self.categorical_cols:
            if col != self.id_column and self.df[col].nunique() < 15:  # Skip ID and columns with too many categories
                selected_cats.append(col)
        
        if selected_cats:
            for col in selected_cats:
                plt.figure(figsize=(12, 6))
                
                # Sort value counts for better visualization
                value_counts = self.df[col].value_counts().sort_values(ascending=False)
                
                # Create the bar chart
                ax = sns.barplot(x=value_counts.index, y=value_counts.values)
                
                # Add value labels on top of bars
                for i, v in enumerate(value_counts.values):
                    ax.text(i, v + 0.1, str(v), ha='center')
                
                plt.title(f'Counts of {col}')
                plt.ylabel('Count')
                plt.xlabel(col)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
        
        return self
    
    def correlation_analysis(self):
        """
        Analyze correlations between numerical variables.
        """
        print("=" * 50)
        print("Correlation Analysis:")
        print("=" * 50)
        
        # Calculate the correlation matrix
        corr_matrix = self.df[self.numerical_cols].corr()
        
        # Print the correlation matrix
        print("Correlation Matrix:")
        print(corr_matrix)
        
        # Visualize the correlation matrix
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
        
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout()
        plt.show()
        
        # Scatter plots for factor scores
        if len(self.factor_scores) >= 2:
            print("\nScatter Plots for Factor Scores:")
            
            # Create a pairplot for factor scores
            sns.pairplot(self.df[self.factor_scores], diag_kind='kde')
            plt.suptitle('Pairwise Relationships Between Factor Scores', y=1.02)
            plt.tight_layout()
            plt.show()
        
        return self
    
    def group_comparisons(self):
        """
        Compare numerical variables across different categorical groups.
        """
        print("=" * 50)
        print("Group Comparisons:")
        print("=" * 50)
        
        # Select categorical variables for grouping (exclude participant_id)
        grouping_vars = [col for col in self.categorical_cols 
                        if col != self.id_column and self.df[col].nunique() < 10]
        
        if not self.factor_scores or not grouping_vars:
            print("Not enough variables for group comparisons.")
            return self
        
        # For each factor score, compare across different categorical groups
        for factor in self.factor_scores:
            for group in grouping_vars:
                # Skip if too many missing values
                if self.df[group].isnull().sum() > 0.5 * len(self.df):
                    continue
                    
                plt.figure(figsize=(12, 6))
                
                # Create violin plot with boxplot inside
                ax = sns.violinplot(x=group, y=factor, data=self.df, inner="box", palette="viridis")
                
                plt.title(f'{factor} by {group}')
                plt.xlabel(group)
                plt.ylabel(factor)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
                
                # Perform ANOVA if there are more than 2 groups, t-test otherwise
                groups = self.df.groupby(group)[factor].apply(list)
                valid_groups = {k: v for k, v in groups.items() if len(v) > 5}  # Require at least 5 samples
                
                if len(valid_groups) >= 2:
                    print(f"\nStatistical test for {factor} across {group} groups:")
                    
                    if len(valid_groups) == 2:
                        # Perform t-test for 2 groups
                        keys = list(valid_groups.keys())
                        t_stat, p_val = stats.ttest_ind(
                            valid_groups[keys[0]], 
                            valid_groups[keys[1]], 
                            equal_var=False  # Welch's t-test (doesn't assume equal variance)
                        )
                        print(f"Independent t-test (Welch's): t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
                        
                    else:
                        # Perform ANOVA for more than 2 groups
                        f_stat, p_val = stats.f_oneway(*valid_groups.values())
                        print(f"One-way ANOVA: F-statistic = {f_stat:.4f}, p-value = {p_val:.4f}")
                    
                    if p_val < 0.05:
                        print(f"The difference in {factor} between {group} groups is statistically significant (p < 0.05).")
                    else:
                        print(f"The difference in {factor} between {group} groups is not statistically significant (p >= 0.05).")
        
        return self
    
    def bmi_analysis(self):
        """
        Analyze the relationship between BMI and factor scores.
        """
        if 'bmi' not in self.df.columns or not self.factor_scores:
            return self
        
        print("=" * 50)
        print("BMI and Factor Scores Analysis:")
        print("=" * 50)
        
        # Scatter plots for BMI vs. each factor score
        for factor in self.factor_scores:
            plt.figure(figsize=(10, 6))
            
            # Create scatter plot with regression line
            sns.regplot(x='bmi', y=factor, data=self.df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            
            # Calculate correlation
            corr, p_val = stats.pearsonr(self.df['bmi'].dropna(), self.df[factor].dropna())
            
            plt.title(f'BMI vs {factor} (Correlation: {corr:.4f}, p-value: {p_val:.4f})')
            plt.xlabel('BMI')
            plt.ylabel(factor)
            plt.tight_layout()
            plt.show()
        
        return self
    
    def perform_pca(self):
        """
        Perform PCA on factor scores to identify patterns.
        """
        # Select factor scores for PCA
        if len(self.factor_scores) < 2:
            print("Not enough factor scores for PCA.")
            return self
        
        print("=" * 50)
        print("Principal Component Analysis (PCA):")
        print("=" * 50)
        
        # Extract factor scores and handle missing values
        X = self.df[self.factor_scores].dropna()
        
        if len(X) < 10:
            print("Too many missing values for PCA.")
            return self
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Print explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        
        print("Explained Variance Ratio:")
        for i, var in enumerate(explained_variance):
            print(f"Component {i+1}: {var:.4f} ({var*100:.2f}%)")
        
        print(f"Total variance explained by 2 components: {sum(explained_variance[:2])*100:.2f}%")
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red')
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.title('Explained Variance by Components')
        plt.tight_layout()
        plt.show()
        
        # Plot PCA results (first two components)
        plt.figure(figsize=(12, 10))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
        
        # Add feature vectors
        for i, factor in enumerate(self.factor_scores):
            plt.arrow(0, 0, pca.components_[0, i]*5, pca.components_[1, i]*5, 
                     color='red', alpha=0.7, head_width=0.1)
            plt.text(pca.components_[0, i]*5.2, pca.components_[1, i]*5.2, factor)
        
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
        plt.title('PCA of Factor Scores')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return self
    
    def parent_education_analysis(self):
        """
        Analyze how parent education levels relate to factor scores.
        """
        if 'parent_1_education' not in self.df.columns or 'parent_2_education' not in self.df.columns:
            return self
        
        if not self.factor_scores:
            return self
        
        print("=" * 50)
        print("Parent Education and Factor Scores Analysis:")
        print("=" * 50)
        
        # Create education level categories if they're not already categorical
        for parent in ['parent_1_education', 'parent_2_education']:
            if self.df[parent].dtype == 'object' and self.df[parent].nunique() < 10:
                # For each factor score, create a boxplot grouped by parent education
                for factor in self.factor_scores:
                    plt.figure(figsize=(14, 8))
                    
                    sns.boxplot(x=parent, y=factor, data=self.df)
                    
                    plt.title(f'{factor} by {parent}')
                    plt.xlabel(parent)
                    plt.ylabel(factor)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.show()
                    
                    # Calculate average factor score by education level
                    avg_by_edu = self.df.groupby(parent)[factor].mean().sort_values()
                    
                    print(f"\nAverage {factor} by {parent}:")
                    print(avg_by_edu)
        
        # Analyze highest education level between both parents
        self.df['highest_parent_edu'] = self.df.apply(
            lambda row: max(str(row['parent_1_education']), str(row['parent_2_education'])) 
            if pd.notna(row['parent_1_education']) and pd.notna(row['parent_2_education']) 
            else row['parent_1_education'] if pd.notna(row['parent_1_education']) 
            else row['parent_2_education'],
            axis=1
        )
        
        if self.df['highest_parent_edu'].nunique() < 10:
            for factor in self.factor_scores:
                plt.figure(figsize=(14, 8))
                
                sns.boxplot(x='highest_parent_edu', y=factor, data=self.df)
                
                plt.title(f'{factor} by Highest Parent Education')
                plt.xlabel('Highest Parent Education')
                plt.ylabel(factor)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
        
        return self
    
    def age_analysis(self):
        """
        Analyze the relationship between age and factor scores.
        """
        if 'age' not in self.df.columns or not self.factor_scores:
            return self
        
        print("=" * 50)
        print("Age and Factor Scores Analysis:")
        print("=" * 50)
        
        # Scatter plots for age vs. each factor score
        for factor in self.factor_scores:
            plt.figure(figsize=(10, 6))
            
            # Create scatter plot with regression line
            sns.regplot(x='age', y=factor, data=self.df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            
            # Calculate correlation
            corr, p_val = stats.pearsonr(self.df['age'].dropna(), self.df[factor].dropna())
            
            plt.title(f'Age vs {factor} (Correlation: {corr:.4f}, p-value: {p_val:.4f})')
            plt.xlabel('Age')
            plt.ylabel(factor)
            plt.tight_layout()
            plt.show()
        
        return self
    
    def study_site_analysis(self):
        """
        Analyze differences between study sites.
        """
        if 'study_site' not in self.df.columns:
            return self
        
        print("=" * 50)
        print("Study Site Analysis:")
        print("=" * 50)
        
        # Count participants per site
        site_counts = self.df['study_site'].value_counts()
        
        plt.figure(figsize=(12, 6))
        site_counts.plot(kind='bar')
        plt.title('Number of Participants per Study Site')
        plt.xlabel('Study Site')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Compare numerical variables across sites
        for col in self.numerical_cols:
            if col not in self.factor_scores and col != 'participant_id':
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='study_site', y=col, data=self.df)
                plt.title(f'{col} by Study Site')
                plt.xlabel('Study Site')
                plt.ylabel(col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
        
        # Compare factor scores across sites
        for factor in self.factor_scores:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='study_site', y=factor, data=self.df)
            plt.title(f'{factor} by Study Site')
            plt.xlabel('Study Site')
            plt.ylabel(factor)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # ANOVA for factor scores across sites
            groups = self.df.groupby('study_site')[factor].apply(list)
            valid_groups = {k: v for k, v in groups.items() if len(v) > 5}
            
            if len(valid_groups) >= 2:
                f_stat, p_val = stats.f_oneway(*valid_groups.values())
                print(f"One-way ANOVA for {factor} across study sites: F-statistic = {f_stat:.4f}, p-value = {p_val:.4f}")
                
                if p_val < 0.05:
                    print(f"The difference in {factor} between study sites is statistically significant (p < 0.05).")
                else:
                    print(f"The difference in {factor} between study sites is not statistically significant (p >= 0.05).")
        
        return self
    
    def run_complete_eda(self):
        """
        Run a complete EDA pipeline on the dataset.
        """
        print("Starting Exploratory Data Analysis...")
        
        # Basic inspection
        self.inspect_data()
        
        # Basic statistics
        self.basic_stats()
        
        # Missing values
        self.missing_values_analysis()
        
        # Distribution analysis
        self.distribution_analysis()
        
        # Categorical visualization
        self.categorical_visualization()
        
        # Correlation analysis
        self.correlation_analysis()
        
        # Group comparisons
        self.group_comparisons()
        
        # BMI analysis
        self.bmi_analysis()
        
        # Age analysis
        self.age_analysis()
        
        # Parent education analysis
        self.parent_education_analysis()
        
        # Study site analysis
        self.study_site_analysis()
        
        # PCA
        self.perform_pca()
        
        print("\nEDA completed successfully!")
        return self
    
    