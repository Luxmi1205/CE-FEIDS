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

# First selecting one validation sample
sample = X_validation[:1]

print("\nSample selected successfully!")
print(f"Sample Shape: {sample.shape}")

# This returns the reconstruction error for each sample.
def anomaly_score(data):

    reconstructed = model.predict(
        data,
        verbose=0)

    errors = np.mean(
        np.square(data - reconstructed),
        axis=1)

    return errors

# Testing the function
sample_score = anomaly_score(
    X_validation[:5]
)
print("\nAnomaly scores generated successfully!")
print(sample_score)

# SHAP explainer for anomaly scores
anomaly_explainer = shap.Explainer(
    anomaly_score,
    background)
print("\nAnomaly SHAP Explainer created successfully!")

anomaly_shap_values = anomaly_explainer(
    sample
)

print("\nAnomaly SHAP values generated successfully!")
print(anomaly_shap_values.values.shape)

# Creating the waterfall plot
plt.figure(figsize=(10, 6))

shap.plots.waterfall(
    anomaly_shap_values[0],
    max_display=10,
    show=False)

plt.savefig(
    "results/explainability/anomaly_score_waterfall.png",
    dpi=300,
    bbox_inches="tight")

plt.close()

print("\nAnomaly Score Waterfall plot saved successfully!")

# absolute SHAP importance
feature_importance = np.abs(
    anomaly_shap_values.values[0])

sorted_indices = np.argsort(
    feature_importance
)[::-1]

# Creating the dataframe
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Importance": feature_importance
})

# Sorting the dataframe
importance_df = importance_df.sort_values(
    by="SHAP Importance",
    ascending=False
)

# displaying the top 10 features
print("\n--------- Top 10 Important Features ----------")
print(importance_df.head(10))

# Saving the table 
importance_df.to_csv(
    "results/explainability/top10_shap_features.csv",
    index=False
)
print("\nTop SHAP feature table saved successfully!")

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

# Now selecting multiple samples
explanation_samples = X_validation[:100]

print("\nGlobal explanation dataset created successfully!")
print(f"Shape: {explanation_samples.shape}")

# This generate Shap values
global_shap_values = anomaly_explainer(
    explanation_samples
)

print("\nGlobal SHAP values generated successfully!")
print(global_shap_values.values.shape)


# Create SHAP Summary Plot
plt.figure(figsize=(10, 8))

shap.summary_plot(
    global_shap_values.values,
    explanation_samples,
    feature_names=feature_names,
    show=False)

plt.savefig(
    "results/explainability/shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nSHAP Summary Plot saved successfully!")

# Creating SHAP Global Bar Plot
plt.figure(figsize=(10, 8))
shap.plots.bar(
    global_shap_values,
    max_display=10,
    show=False
)
plt.savefig(
    "results/explainability/shap_bar_plot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nSHAP Bar Plot saved successfully!")