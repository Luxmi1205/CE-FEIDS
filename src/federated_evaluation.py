import joblib
import os
import numpy as np

from tensorflow.keras.models import load_model
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Loading the federated model
global_model = load_model(
    "models/federated/global_autoencoder.keras")

print("Federated global model loaded successfully!")

# Loading dataset
X_validation = joblib.load(
    "data/processed/X_validation_scaled.pkl"
)

y_validation = joblib.load(
    "data/processed/y_validation.pkl")

print("\nValidation dataset loaded successfully!")

# Converting all the labels to binary
y_validation_binary = (
    y_validation != "BenignTraffic"
).astype(int)

print("\nBinary labels created successfully!")

# Generating reconstructed samples
X_validation_reconstructed = global_model.predict(
    X_validation
)

print("Validation samples reconstructed successfully!")

# Calculating reconstruction errors
validation_errors = np.mean(
    np.square(X_validation - X_validation_reconstructed),
    axis=1)

print("Validation reconstruction errors calculated successfully!")
print(validation_errors.shape)

os.makedirs( # saving reconstruction error
    "results/reconstruction",
    exist_ok=True)
joblib.dump(
    validation_errors,
    "results/reconstruction/federated_validation_errors.pkl"
)

print("Federated validation reconstruction errors saved successfully!")

# Load threshold
with open("results/metrics/threshold.txt", "r") as file:
    threshold = float(file.read())

print(f"\nThreshold Loaded: {threshold:.8f}")

# Thi generates predictions
validation_predictions = (
    validation_errors > threshold
).astype(int)

print("Validation predictions generated successfully!")

print(validation_predictions.shape)

print("\nFederated Reconstruction Error Statistics")
print(f"Minimum : {validation_errors.min():.8f}")
print(f"Mean    : {validation_errors.mean():.8f}")
print(f"Maximum : {validation_errors.max():.8f}")
print(f"\nThreshold : {threshold:.8f}")


# For calculatig evaluation metrix
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

print("\nFederated evaluation completed successfully!")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)


# Creating metrics folder
import os
os.makedirs(
    "results/metrics",
    exist_ok=True)

# Saving federated evaluation metrics
with open("results/metrics/federated_metrics.txt", "w") as file:

    file.write("Federated Autoencoder Evaluation\n")
    file.write("=" * 35 + "\n")

    file.write(f"Threshold : {threshold:.8f}\n")
    file.write(f"Accuracy  : {accuracy:.4f}\n")
    file.write(f"Precision : {precision:.4f}\n")
    file.write(f"Recall    : {recall:.4f}\n")
    file.write(f"F1 Score  : {f1:.4f}\n")
    file.write(f"ROC-AUC   : {roc_auc:.4f}\n")

print("Federated metrics saved successfully!")

federated_confusion_df = pd.DataFrame(
    conf_matrix,
    index=["Actual Benign", "Actual Attack"],
    columns=["Predicted Benign", "Predicted Attack"]
)

federated_confusion_df.to_csv(
    "results/metrics/federated_confusion_matrix.csv"
)

print("Federated confusion matrix saved successfully!")