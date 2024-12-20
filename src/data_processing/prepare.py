import pandas as pd
import os

# Define file paths
red_wine_path = 'data/raw/winequality-red.csv'
white_wine_path = 'data/raw/winequality-white.csv'
processed_data_path = 'data/processed/prepared_data.csv'

# Ensure the output directory exists
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

# Function to load and clean data
def load_and_clean_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Drop any rows with missing values (if any)
    df.dropna(inplace=True)
    
    # You can add more cleaning steps here as needed
    # For example, converting column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Return the cleaned dataframe
    return df

# Load and clean the red wine dataset
red_wine_df = load_and_clean_data(red_wine_path)

# Load and clean the white wine dataset
white_wine_df = load_and_clean_data(white_wine_path)

# Combine both datasets into one (you can modify this if needed)
combined_df = pd.concat([red_wine_df, white_wine_df], ignore_index=True)

# Save the combined cleaned dataset to a single file
combined_df.to_csv(processed_data_path, index=False)

print("Data processing complete. Combined dataset saved as prepared_data.csv.")
