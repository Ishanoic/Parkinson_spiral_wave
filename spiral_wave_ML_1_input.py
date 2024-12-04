# Import necessary packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from skimage import feature
import cv2
import pickle
import os
import argparse
import numpy as np

def quantify_image(image):
    # Compute HOG features
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read the image at {image_path}.")
    
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image
    image = cv2.resize(image, (200, 200))
    # Apply binary thresholding
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Extract features
    features = quantify_image(image)
    return features

def main():
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Path to the trained model")
    ap.add_argument("-l", "--labels", required=True, help="Path to the label encoder")
    ap.add_argument("-s", "--spiral", required=True, help="Path to the spiral input image")
    ap.add_argument("-w", "--wave", required=True, help="Path to the wave input image")
    args = vars(ap.parse_args())

    # Load the model and label encoder
    print("[INFO] Loading model and label encoder...")
    with open(args["model"], "rb") as f:
        model = pickle.load(f)
    with open(args["labels"], "rb") as f:
        le = pickle.load(f)

    # Preprocess the spiral and wave images
    print("[INFO] Preprocessing the input spiral image...")
    spiral_features = preprocess_image(args["spiral"])
    print(f"Spiral features shape: {spiral_features.shape}")

    print("[INFO] Preprocessing the input wave image...")
    wave_features = preprocess_image(args["wave"])
    print(f"Wave features shape: {wave_features.shape}")

    # Make a prediction for spiral image
    print("[INFO] Making prediction for spiral image...")
    spiral_pred = model.predict([spiral_features])[0]
    spiral_label = le.inverse_transform([spiral_pred])[0]

    # Make a prediction for wave image
    print("[INFO] Making prediction for wave image...")
    wave_pred = model.predict([wave_features])[0]
    wave_label = le.inverse_transform([wave_pred])[0]

    # Display the result
    print(f"[RESULT] The spiral image is classified as: {spiral_label}")
    print(f"[RESULT] The wave image is classified as: {wave_label}")

if __name__ == "__main__":
    main()



# run - python spiral_wave_ML_1_input.py --model spiral_wave_model.pkl --labels label_encoder.pkl --spiral parkinson-unhealthy.png --wave drawings\wave\testing\parkinson\V03PO04.png
