import streamlit as st
import cv2
import numpy as np
import pickle
from skimage import feature

# Function to extract HOG features from an image
def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

# Function to preprocess the input image
def preprocess_image(image):
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize image
    image = cv2.resize(image, (200, 200))
    # Apply binary thresholding
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Extract features
    features = quantify_image(image)
    return features

# Streamlit app
def main():
    st.title("Parkinson's Prediction using Spiral and Wave Images")
    st.markdown("This app predicts Parkinson's based on spiral and wave images.")

    # Upload images
    spiral_image = st.file_uploader("Upload Spiral Image", type=["png", "jpg", "jpeg"])
    wave_image = st.file_uploader("Upload Wave Image", type=["png", "jpg", "jpeg"])

    # Load the trained model and label encoder
    with open("spiral_wave_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    if spiral_image is not None and wave_image is not None:
        # Read images as numpy arrays
        spiral_img = np.asarray(bytearray(spiral_image.read()), dtype=np.uint8)
        wave_img = np.asarray(bytearray(wave_image.read()), dtype=np.uint8)

        spiral_img = cv2.imdecode(spiral_img, 1)
        wave_img = cv2.imdecode(wave_img, 1)

        # Process the images
        spiral_features = preprocess_image(spiral_img)
        wave_features = preprocess_image(wave_img)

        # Make prediction for spiral image
        spiral_pred = model.predict([spiral_features])[0]
        spiral_label = le.inverse_transform([spiral_pred])[0]

        # Make prediction for wave image
        wave_pred = model.predict([wave_features])[0]
        wave_label = le.inverse_transform([wave_pred])[0]

        # Display the images and results
        st.image(spiral_img, caption="Spiral Image", use_container_width=True)
        st.image(wave_img, caption="Wave Image", use_container_width=True)
        st.write(f"The spiral image is classified as: {spiral_label}")
        st.write(f"The wave image is classified as: {wave_label}")

if __name__ == "__main__":
    main()

#run - streamlit run spiral_wave_app.py
