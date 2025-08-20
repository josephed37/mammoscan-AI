import cv2
import os
from tqdm import tqdm # For a nice progress bar

def process_and_save_image(read_path, save_path, target_size=(224, 224)):
    """
    Reads an image, resizes it to a standard size, and saves it.
    """
    try:
        img = cv2.imread(read_path)
        if img is None:
            print(f"Warning: Could not read image {read_path}. Skipping.")
            return

        # Resize the image
        img_resized = cv2.resize(img, target_size)
        
        # Save the processed image
        cv2.imwrite(save_path, img_resized)

    except Exception as e:
        print(f"Error processing {read_path}: {e}")

def process_dataframe(df, save_dir):
    """
    Processes all images listed in a DataFrame and saves them into class folders.
    """
    # Create the main save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Using tqdm for a nice progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing images for {os.path.basename(save_dir)}"):
        label = row['label']
        filepath = row['filepath']
        
        # Create a subdirectory for the label (e.g., .../train/Cancer)
        label_dir = os.path.join(save_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Define the full save path for the new image
        save_path = os.path.join(label_dir, os.path.basename(filepath))
        
        # Process and save the image using our other function
        process_and_save_image(filepath, save_path)