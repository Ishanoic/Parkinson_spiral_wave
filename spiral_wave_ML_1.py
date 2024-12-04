# Add your existing imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import pickle

def quantify_image(image):
    # Extract HOG features from the image
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

def load_split(path):
    imagePaths = list(paths.list_images(path))  # Get all images
    data = []
    labels = []

    # Process each image
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]  # Extract label (healthy/parkinson)
        image = cv2.imread(imagePath)  # Load image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        image = cv2.resize(image, (200, 200))  # Resize image to fixed size

        # Threshold image
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Extract features from image
        features = quantify_image(image)
        data.append(features)  # Add features to data
        labels.append(label)  # Add label to labels

    return (np.array(data), np.array(labels))

# Update argument parsing to specify dataset location
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-t", "--trials", type=int, default=5, help="# of trials to run")
args = vars(ap.parse_args())

# Define paths to both spiral and wave datasets
trainingPath_spiral = os.path.sep.join([args["dataset"], "spiral", "training"])
testingPath_spiral = os.path.sep.join([args["dataset"], "spiral", "testing"])

trainingPath_wave = os.path.sep.join([args["dataset"], "wave", "training"])
testingPath_wave = os.path.sep.join([args["dataset"], "wave", "testing"])

# Load spiral training and testing data
print("[INFO] loading spiral data...")
(trainX_spiral, trainY_spiral) = load_split(trainingPath_spiral)
(testX_spiral, testY_spiral) = load_split(testingPath_spiral)

# Load wave training and testing data
print("[INFO] loading wave data...")
(trainX_wave, trainY_wave) = load_split(trainingPath_wave)
(testX_wave, testY_wave) = load_split(testingPath_wave)

# Combine data and labels from spiral and wave datasets
trainX = np.vstack([trainX_spiral, trainX_wave])  # Combine spiral and wave features
trainY = np.hstack([trainY_spiral, trainY_wave])  # Combine corresponding labels
testX = np.vstack([testX_spiral, testX_wave])  # Same for testing data
testY = np.hstack([testY_spiral, testY_wave])

# Encode labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)


# Initialize trial dictionary to store metrics
trials = {}

# Loop over trials to train the model
for i in range(0, args["trials"]):
    print(f"[INFO] training model {i+1} of {args['trials']}...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(trainX, trainY)

    predictions = model.predict(testX)
    metrics = {}

    cm = confusion_matrix(testY, predictions).flatten()
    (tn, fp, fn, tp) = cm
    metrics["acc"] = (tp + tn) / float(cm.sum())  # accuracy
    metrics["sensitivity"] = tp / float(tp + fn)  # sensitivity
    metrics["specificity"] = tn / float(tn + fp)  # specificity

    for (k, v) in metrics.items():
        l = trials.get(k, [])
        l.append(v)
        trials[k] = l

# Display metrics for the trials
for metric in ("acc", "sensitivity", "specificity"):
    values = trials[metric]
    mean = np.mean(values)
    std = np.std(values)
    print(f"{metric} - u={mean:.4f}, o={std:.4f}")

# Save the model and label encoder
print("[INFO] Saving model and label encoder...")
with open("spiral_wave_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("[INFO] Model and label encoder saved!")

# Montage creation
# Get testing image paths from both spiral and wave datasets
testingPaths_spiral = list(paths.list_images(testingPath_spiral))
testingPaths_wave = list(paths.list_images(testingPath_wave))
testingPaths = testingPaths_spiral + testingPaths_wave  # Combine both lists

# Randomly select 25 images
idxs = np.arange(0, len(testingPaths))
idxs = np.random.choice(idxs, size=(25,), replace=False)  # Random selection of 25 images
images = []

# Process and predict on testing images
for i in idxs:
    image = cv2.imread(testingPaths[i])  # Read image
    output = image.copy()
    output = cv2.resize(output, (128, 128))  # Resize for display

    # Pre-process the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))  # Resize for HOG feature extraction
    image = cv2.threshold(image, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # Threshold image

    # Extract HOG features and predict
    features = quantify_image(image)
    preds = model.predict([features])
    label = le.inverse_transform(preds)[0]

    # Draw label on image
    color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    images.append(output)  # Add processed image to the list

# Create and display the montage
montage = build_montages(images, (128, 128), (5, 5))[0]  # Create montage with 5x5 grid
cv2.imshow("Output", montage)  # Show the montage
cv2.waitKey(0)  # Wait for a key press to close the window


# run with: python spiral_wave_ML_1.py --dataset drawings --trials 5


# the script randomly selects 25 images from the testing set and performs predictions.



