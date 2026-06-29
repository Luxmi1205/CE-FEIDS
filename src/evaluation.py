import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load training history
history = joblib.load("models/autoencoder/training_history.pkl")
print("Training history loaded successfully!")
print(history.keys())


# This plot Training and Validation Loss
plt.figure(figsize=(8, 5))

plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")

plt.title("Autoencoder Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("results/plots/training_history.png")

plt.show()

print("Training history plot saved successfully!")

# Loading reconstruction errors
train_errors = joblib.load(
    "results/reconstruction/train_errors.pkl"
)

print("Training reconstruction errors loaded successfully!")

print(train_errors.shape)

# Calculate Mean and Standard Deviation
mean_error = np.mean(train_errors)
std_error = np.std(train_errors)

# Mean + 3 Standard Deviations
threshold = mean_error + (3 * std_error)

print(f"\nMean Reconstruction Error: {mean_error:.8f}")
print(f"Standard Deviation: {std_error:.8f}")
print(f"Threshold (Mean + 3σ): {threshold:.8f}")

# this creates the metrics folder
os.makedirs("results/metrics", exist_ok=True)

# Saving threshold
with open("results/metrics/threshold.txt", "w") as file:
    file.write(f"{threshold:.8f}")

print("\nThreshold saved successfully!")

# Load validation and test labels
y_validation = joblib.load("data/processed/y_validation.pkl")
y_test = joblib.load("data/processed/y_test.pkl")

# Convert labels to binary
y_validation_binary = (y_validation != "BenignTraffic").astype(int)
y_test_binary = (y_test != "BenignTraffic").astype(int)

print("\nBinary labels created successfully!")

print("Validation Label Counts:") #to verify
print(y_validation_binary.value_counts())

print("\nTest Label Counts:")
print(y_test_binary.value_counts())

# Loading validation reconstruction errors
validation_errors = joblib.load(
    "results/reconstruction/validation_errors.pkl"
)

print("\nValidation reconstruction errors loaded successfully!")

# Generate predictions
def predict_anomalies(errors, threshold):
    return (errors > threshold).astype(int)#this converts boolean array into the format required for evaluation

print("Validation predictions generated successfully!")

print("\nValidation Predictions Shape:") #to verify

validation_predictions = predict_anomalies(validation_errors, threshold)

print(validation_predictions.shape)

# Calculating evaluation metrics
accuracy = accuracy_score(
    y_validation_binary,
    validation_predictions
)

precision = precision_score(
    y_validation_binary,
    validation_predictions
)

recall = recall_score(
    y_validation_binary,
    validation_predictions
)

f1 = f1_score(
    y_validation_binary,
    validation_predictions
)

roc_auc = roc_auc_score(
    y_validation_binary,
    validation_predictions
)

conf_matrix = confusion_matrix(
    y_validation_binary,
    validation_predictions
)

print("\nEvaluation metrics calculated successfully!")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

with open("results/metrics/baseline_metrics.txt", "w") as file:
    file.write("Baseline Autoencoder Evaluation\n")
    file.write("=" * 35 + "\n\n")

    file.write(f"Threshold : {threshold:.8f}\n")
    file.write(f"Accuracy  : {accuracy:.4f}\n")
    file.write(f"Precision : {precision:.4f}\n")
    file.write(f"Recall    : {recall:.4f}\n")
    file.write(f"F1 Score  : {f1:.4f}\n")
    file.write(f"ROC-AUC   : {roc_auc:.4f}\n")

print("Baseline metrics saved successfully!")

confusion_df = pd.DataFrame(
    conf_matrix,
    index=["Actual Benign", "Actual Attack"],
    columns=["Predicted Benign", "Predicted Attack"]
)

confusion_df.to_csv(
    "results/metrics/confusion_matrix.csv"
)

print("Confusion matrix saved successfully!")