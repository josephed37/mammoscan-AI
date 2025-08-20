import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Path Setup ---
# This is the reliable way to find the project's root directory
# It works regardless of where you run the script from.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Custom Modules ---
from ml.src.data_utils import load_image_paths
from ml.src.preprocess_utils import process_dataframe

# --- Constants ---
RAW_DATA_DIR = os.path.join(project_root, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

def main():
    """Main function to run the data preprocessing pipeline."""
    print("🚀 Starting data preprocessing pipeline...")

    # Step 1: Load Data Paths
    df = load_image_paths(RAW_DATA_DIR)

    # Step 2: Ensure 'filepath' column is string type
    df['filepath'] = df['filepath'].astype(str)

    # Step 3: Separate Original and Augmented data
    # Make sure these strings exactly match your folder names
    original_df = df[df['filepath'].str.contains('Original Dataset')]
    augmented_df = df[df['filepath'].str.contains('Augmented Dataset')]

    # Step 4: Split the ORIGINAL dataset into train, validation, and test sets
    train_df, test_df = train_test_split(
        original_df,
        test_size=TEST_SIZE,
        stratify=original_df['label'],
        random_state=RANDOM_STATE
    )
    
    relative_val_size = VAL_SIZE / (1 - TEST_SIZE)
    train_df, val_df = train_test_split(
        train_df,
        test_size=relative_val_size,
        stratify=train_df['label'],
        random_state=RANDOM_STATE
    )

    # Step 5: Add the AUGMENTED data ONLY to the training set
    final_train_df = pd.concat([train_df, augmented_df], ignore_index=True)
    
    print("\nDataset split sizes:")
    print(f"  Training set:   {len(final_train_df)} images")
    print(f"  Validation set: {len(val_df)} images")
    print(f"  Test set:       {len(test_df)} images\n")

    # Step 6: Process and save each dataset split
    process_dataframe(final_train_df, os.path.join(PROCESSED_DATA_DIR, 'train'))
    process_dataframe(val_df, os.path.join(PROCESSED_DATA_DIR, 'val'))
    process_dataframe(test_df, os.path.join(PROCESSED_DATA_DIR, 'test'))

    print("\n✅ Data preprocessing pipeline finished successfully!")

if __name__ == '__main__':
    main()