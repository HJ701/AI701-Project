import os
import cv2
import numpy as np
from PIL import Image

def ingest_images(data_path):
    """
    Reads images from the specified data path and classifies them as intraoral
    images based on average saturation greater than 10.

    Args:
        data_path (str): The path to the data directory.

    Returns:
        dict: A dictionary containing intraoral image file names and their PIL Image objects.
    """
    intraoral_images = {}
    files = os.listdir(data_path)

    for file in files:
        # Check if file is an image
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            img_path = os.path.join(data_path, file)
            try:
                # Read the image using OpenCV
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    print(f"Error reading image {file} with OpenCV.")
                    continue

                # Convert BGR to HSV color space
                hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

                # Calculate average saturation
                avg_saturation = hsv_img[:, :, 1].mean()

                # Print metrics for each image
                print(f"{file} - Avg Saturation: {avg_saturation:.2f}")

                # Classify image based on threshold
                if avg_saturation > 10:
                    # Likely an intraoral image
                    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                    intraoral_images[file] = img_pil
                else:
                    # Likely a radiograph
                    pass  # You can handle radiographs here if needed

            except Exception as e:
                print(f"Error processing image {file}: {e}")

    return intraoral_images
