"""
Script to randomly sample 40% of the dataset for testing.
Generates FAKE random labels (not the actual labels from the CSV).
Outputs a CSV with 'filename' and 'label' columns.
"""

import pandas as pd
import numpy as np

# Configuration
INPUT_FILE = "/Volumes/Harshas ssd/development/ieee-ml-hack-data/archive/test_labels.csv"
OUTPUT_FILE = "test_sample_40pct_fake_labels.csv"
SAMPLE_FRACTION = 0.40  # 40% of the dataset
RANDOM_STATE = 42  # For reproducibility (change or remove for different samples)

def main():
    # Read the dataset
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Original dataset size: {len(df)} rows")
    
    # Sample 40% randomly (only filenames)
    sampled_df = df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE)
    
    # Generate FAKE random labels (0 or 1)
    np.random.seed(RANDOM_STATE)
    fake_labels = np.random.randint(0, 2, size=len(sampled_df))
    
    # Create output dataframe with filename and fake labels
    output_df = pd.DataFrame({
        'filename': sampled_df['filename'].values,
        'label': fake_labels
    })
    
    print(f"Sampled dataset size: {len(output_df)} rows")
    print(f"Fake label distribution:\n{output_df['label'].value_counts()}\n")
    
    # Save to CSV
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Display first 10 rows
    print("\nFirst 10 rows of sampled data with fake labels:")
    print(output_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
