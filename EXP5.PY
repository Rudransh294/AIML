import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import data from the previous example
def import_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data imported successfully.")
        return df
    except Exception as e:
        print(f"Error importing data: {e}")
        return None

# EDA Function
def exploratory_data_analysis(df):
    if df is None:
        print("No data available for analysis.")
        return

    # 1. General Info
    print("\n--- General Information ---")
    print(df.info())  # Show data types, non-null counts, memory usage
    
    # 2. Basic Statistics
    print("\n--- Descriptive Statistics ---")
    print(df.describe())  # Summary statistics for numerical columns
    
    # 3. Missing Data Analysis
    print("\n--- Missing Data ---")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Missing values percentage
    total_cells = df.size
    total_missing = missing_values.sum()
    print(f"\nPercentage of missing data: {(total_missing / total_cells) * 100:.2f}%")
    
    # Visualize missing data using a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title('Missing Data Heatmap')
    plt.show()
    
    # 4. Distribution of Numerical Columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns].hist(bins=15, figsize=(15, 6), layout=(2, len(numerical_columns)//2 + 1))
    plt.tight_layout()
    plt.show()

    # 5. Correlation Analysis
    print("\n--- Correlation Matrix ---")
    correlation = df.corr()
    print(correlation)

    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()
    
    # 6. Checking Categorical Variables
    print("\n--- Categorical Features Summary ---")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        print(f"{col}:\n{df[col].value_counts()}\n")

# Example usage
if __name__ == "__main__":
    # Provide the file path to import the dataset
    file_path = '/absolute/path/to/example_data.csv'  # Replace this with your actual path
    
    # Import the dataset
    df = import_data(file_path)
    
    # Perform EDA on the dataset
    exploratory_data_analysis(df)
