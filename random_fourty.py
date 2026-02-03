import pandas as pd
import os

def process_labels():
    input_path = '/Volumes/Harshas ssd/development/ieee-ml-hack-data/archive/test_labels.csv'
    output_path = 'percentage_40.csv'

    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    try:
        # Read the CSV file
        df = pd.read_csv(input_path)
        print(f"Successfully loaded {len(df)} rows from {input_path}")
        print(f"Columns: {df.columns.tolist()}")

        # Ensure 'label' column exists (case-insensitive check)
        label_col = next((col for col in df.columns if col.lower() == 'label'), 'label')
        
        if label_col not in df.columns:
            print(f"Error: '{label_col}' column not found in CSV.")
            return

        # Filter data for labels 1 and 0
        ones = df[df[label_col] == 1]
        zeros = df[df[label_col] == 0]

        print(f"Count of label 1: {len(ones)}")
        print(f"Count of label 0: {len(zeros)}")

        # Check if we have enough samples
        total_samples = len(df)
        if total_samples == 0:
            print("Error: No samples in the dataset.")
            return

        # Sample 20% of total from each group
        n_ones = min(int(0.2 * total_samples), len(ones))
        n_zeros = min(int(0.2 * total_samples), len(zeros))
        
        if n_ones == 0 and len(ones) > 0:
            n_ones = 1  # at least 1 if available
        if n_zeros == 0 and len(zeros) > 0:
            n_zeros = 1
        
        if n_ones == 0 or n_zeros == 0:
            print("Error: Not enough samples for 20% of total sampling.")
            return

        sampled_ones = ones.sample(n=n_ones)
        sampled_zeros = zeros.sample(n=n_zeros)

        # Combine the samples
        result = pd.concat([sampled_ones, sampled_zeros])

        # Save to a new CSV file
        result.to_csv(output_path, index=False)
        print(f"Success! Compiled {len(result)} random samples ({n_ones} from label 1, {n_zeros} from label 0, total {int(0.4 * total_samples)} expected) into '{output_path}'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    process_labels()