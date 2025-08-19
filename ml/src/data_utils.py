import os
import pandas as pd

def load_image_paths(data_path):
    """
    Scans the data directory and returns a DataFrame with image paths and labels.
    """
    filepaths = []
    labels = []
    
    # Get all subdirectories (e.g., Original/Cancer, Augmented/Non-Cancer)
    for dirpath, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(dirpath, filename)
                filepaths.append(filepath)
                
                # Extract the label (e.g., 'Cancer', 'Non-Cancer') from the folder name
                label = os.path.basename(os.path.dirname(filepath))
                labels.append(label)
                
    # Create a DataFrame
    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    return df