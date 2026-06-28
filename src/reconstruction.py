import joblib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Loading trained Autoencoder
autoencoder = load_model("models/autoencoder/best_model.keras")

print("Autoencoder loaded successfully!")

# Load processed datasets
X_train = joblib.load("data/processed/X_train_benign_scaled.pkl")

X_validation = joblib.load("data/processed/X_validation_scaled.pkl")
X_test = joblib.load("data/processed/X_test_scaled.pkl")

y_validation = joblib.load("data/processed/y_validation.pkl")
y_test = joblib.load("data/processed/y_test.pkl")

print("Processed datasets loaded successfully!")

# Generating reconstructed samples
train_reconstructed = autoencoder.predict(X_train, verbose=1)

validation_reconstructed = autoencoder.predict(
    X_validation,
    verbose=1
)

test_reconstructed = autoencoder.predict(
    X_test,
    verbose=1
)

print("\nReconstructed samples generated successfully!")

# Calculating our reconstruction errors (Mean Squared Error per sample)
train_errors = np.mean(
    np.square(X_train - train_reconstructed),
    axis=1
)

validation_errors = np.mean(
    np.square(X_validation - validation_reconstructed),
    axis=1
)

test_errors = np.mean(
    np.square(X_test - test_reconstructed),
    axis=1
)

print("\nReconstruction errors calculated successfully!")

print(f"Training Errors Shape: {train_errors.shape}")
print(f"Validation Errors Shape: {validation_errors.shape}")
print(f"Test Errors Shape: {test_errors.shape}")

# Create results folder if it doesn't exist
os.makedirs("results/reconstruction", exist_ok=True)

# Saving reconstruction errors
joblib.dump(
    train_errors,
    "results/reconstruction/train_errors.pkl"
)

joblib.dump(
    validation_errors,
    "results/reconstruction/validation_errors.pkl"
)

joblib.dump(
    test_errors,
    "results/reconstruction/test_errors.pkl"
)

print("\nReconstruction errors saved successfully!")