import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model

# Loading the federated model
model = load_model(
    "models/federated/global_autoencoder.keras"
)

print("Federated model loaded successfully!")

# Loading the validation dataset
X_validation = joblib.load(
    "data/processed/X_validation_scaled.pkl"
)

print("Validation dataset loaded successfully!")
print(f"Validation Shape: {X_validation.shape}")

# Loading the features name
feature_names = joblib.load(
    "data/processed/feature_names.pkl"
)

print("Feature names loaded successfully!")
print(f"Number of Features: {len(feature_names)}")
print("\nFirst 5 Features:")
print(feature_names[:5])

# Loading the benign training data
X_train_benign = joblib.load(
    "data/processed/X_train_benign_scaled.pkl"
)

print("\nBenign training dataset loaded successfully!")
print(f"Shape: {X_train_benign.shape}")

# Selecting background samples for SHAP
background = X_train_benign[:100]

print("\nBackground dataset created successfully!")
print(f"Background Shape: {background.shape}")

# Creating the SHAP explainer
explainer = shap.Explainer(
    model,
    background
)

print("\nSHAP Explainer created successfully!")

os.makedirs( # to save results
    "results/explainability",
    exist_ok=True
)

# First selecting one validation sample
sample = X_validation[:1]

print("\nSample selected successfully!")
print(f"Sample Shape: {sample.shape}")

# Generating SHap values
shap_values = explainer(sample)

print("\nSHAP values generated successfully!")

print("\nSHAP Values Shape:") # to verify SHAP value
print(shap_values.values.shape)

# Ploting SHAP explanation for the first output features
plt.figure(figsize=(10, 6))

shap.plots.waterfall(
    shap_values[0, :, 0],
    max_display=10,
    show=False)

plt.savefig(
    "results/explainability/shap_waterfall_output0.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nSHAP Waterfall plot saved successfully!")