"""
Script to generate random predictions for test_public.csv
Adds 'label' (0 or 1) and 'probability' (0.00 to 1.00) columns.
"""

import pandas as pd
import numpy as np

# Configuration
INPUT_FILE = "/Volumes/Harshas ssd/development/ieee-ml-hack-data/archive/test_public.csv"
OUTPUT_FILE = "test_predictions.csv"
RANDOM_STATE = 42  # For reproducibility

def main():
    # Read the dataset
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Original dataset size: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}\n")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_STATE)
    
    # Generate random labels (0 or 1)
    random_labels = np.random.randint(0, 2, size=len(df))
    
    # Generate random probabilities (0.00 to 1.00)
    random_probabilities = np.random.uniform(0.0, 1.0, size=len(df))
    
    # Add the new columns
    df['label'] = random_labels
    df['probability'] = random_probabilities.round(2)  # Round to 2 decimal places
    
    print(f"Generated {len(df)} predictions")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")
    print(f"Probability stats:")
    print(f"  Min: {df['probability'].min():.2f}")
    print(f"  Max: {df['probability'].max():.2f}")
    print(f"  Mean: {df['probability'].mean():.2f}\n")
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Display first 10 rows
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
