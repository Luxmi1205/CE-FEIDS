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

print("\nTraining history plot saved successfully!")

# Loading reconstruction errors
train_errors = joblib.load(
    "results/reconstruction/train_errors.pkl"
)

print("\nTraining reconstruction errors loaded successfully!")

print(train_errors.shape)

# Calculate Mean and Standard Deviation
mean_error = np.mean(train_errors)
std_error = np.std(train_errors)

# Mean + 3 Standard Deviations
threshold_mean3sigma = mean_error + (3 * std_error)

#..........................................95th Percentile Threshold....................................
threshold_95 = np.percentile(train_errors, 95)

#..........................................99th Percentile Threshold....................................
threshold_99 = np.percentile(train_errors, 99)

print(f"\nMean Reconstruction Error: {mean_error:.8f}")
print(f"Standard Deviation: {std_error:.8f}")
print(f"Threshold (Mean + 3σ): {threshold_mean3sigma:.8f}")

# this creates the metrics folder
os.makedirs("results/metrics", exist_ok=True)

# Saving threshold
with open("results/metrics/threshold.txt", "w") as file:
    file.write(f"{threshold_mean3sigma:.8f}")

print("\nThreshold saved successfully!")

# Load validation and test labels
y_validation = joblib.load("data/processed/y_validation.pkl")
y_test = joblib.load("data/processed/y_test.pkl")

# Convert labels to binary
y_validation_binary = (y_validation != "BenignTraffic").astype(int)
y_test_binary = (y_test != "BenignTraffic").astype(int)

print("\nBinary labels created successfully!")

print("\nValidation Label Counts:") #to verify
print(y_validation_binary.value_counts())

print("\nTest Label Counts:")
print(y_test_binary.value_counts())

# Loading validation reconstruction errors
validation_errors = joblib.load(
    "results/reconstruction/validation_errors.pkl"
)
print("\nValidation reconstruction errors loaded successfully!")

# Testing validation reconstruction errors
test_errors = joblib.load(
    "results/reconstruction/test_errors.pkl"
)
print("\nTest reconstruction errors loaded successfully!")

# Generate predictions
def predict_anomalies(errors, threshold):
    return (errors > threshold).astype(int)#this converts boolean array into the format required for evaluation

validation_predictions_mean3sigma = predict_anomalies(validation_errors, threshold_mean3sigma)

validation_predictions_95 = predict_anomalies(
    validation_errors,
    threshold_95
)

validation_predictions_99 = predict_anomalies(
    validation_errors,
    threshold_99
)

print("\nValidation predictions generated successfully!")

print("\nValidation Predictions Shape:") #to verify
print(validation_predictions_mean3sigma.shape)
print(validation_predictions_99.shape)

# Calculating evaluation metrics
accuracy = accuracy_score(
    y_validation_binary,
    validation_predictions_mean3sigma
)

precision = precision_score(
    y_validation_binary,
    validation_predictions_mean3sigma
)

recall = recall_score(
    y_validation_binary,
    validation_predictions_mean3sigma
)

f1 = f1_score(
    y_validation_binary,
    validation_predictions_mean3sigma
)

roc_auc = roc_auc_score(
    y_validation_binary,
    validation_predictions_mean3sigma
)

conf_matrix = confusion_matrix(
    y_validation_binary,
    validation_predictions_mean3sigma
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

    file.write(f"Threshold : {threshold_mean3sigma:.8f}\n")
    file.write(f"Accuracy  : {accuracy:.4f}\n")
    file.write(f"Precision : {precision:.4f}\n")
    file.write(f"Recall    : {recall:.4f}\n")
    file.write(f"F1 Score  : {f1:.4f}\n")
    file.write(f"ROC-AUC   : {roc_auc:.4f}\n")

print("\nBaseline metrics saved successfully!")

confusion_df = pd.DataFrame(
    conf_matrix,
    index=["Actual Benign", "Actual Attack"],
    columns=["Predicted Benign", "Predicted Attack"]
)

confusion_df.to_csv(
    "results/metrics/confusion_matrix.csv"
)

print("Confusion matrix saved successfully!")

#..........................................95th Percentile Threshold....................................
print(f"\n95th Percentile Threshold: {threshold_95:.8f}")

accuracy_95 = accuracy_score(
    y_validation_binary,
    validation_predictions_95
)

precision_95 = precision_score(
    y_validation_binary,
    validation_predictions_95
)

recall_95 = recall_score(
    y_validation_binary,
    validation_predictions_95
)

f1_95 = f1_score(
    y_validation_binary,
    validation_predictions_95
)

roc_auc_95 = roc_auc_score(
    y_validation_binary,
    validation_predictions_95
)

print("\n========== 95th Percentile Results ==========")

print(f"Accuracy : {accuracy_95:.4f}")
print(f"Precision: {precision_95:.4f}")
print(f"Recall   : {recall_95:.4f}")
print(f"F1 Score : {f1_95:.4f}")
print(f"ROC-AUC  : {roc_auc_95:.4f}")


#..........................................99th Percentile Threshold....................................
print(f"\n99th Percentile Threshold: {threshold_99:.8f}")

accuracy_99 = accuracy_score(
    y_validation_binary,
    validation_predictions_99
)

precision_99 = precision_score(
    y_validation_binary,
    validation_predictions_99
)

recall_99 = recall_score(
    y_validation_binary,
    validation_predictions_99
)

f1_99 = f1_score(
    y_validation_binary,
    validation_predictions_99
)

roc_auc_99 = roc_auc_score(
    y_validation_binary,
    validation_predictions_99
)

print("\n========== 99th Percentile Results ==========")

print(f"Accuracy : {accuracy_99:.4f}")
print(f"Precision: {precision_99:.4f}")
print(f"Recall   : {recall_99:.4f}")
print(f"F1 Score : {f1_99:.4f}")
print(f"ROC-AUC  : {roc_auc_99:.4f}")


# Testing predictions
test_predictions = predict_anomalies(
    test_errors,
    threshold_mean3sigma
)

print("\nTest predictions generated successfully!")
print(test_predictions.shape)

# Calculate test evaluation metrics
test_accuracy = accuracy_score(
    y_test_binary,
    test_predictions
)

test_precision = precision_score(
    y_test_binary,
    test_predictions
)

test_recall = recall_score(
    y_test_binary,
    test_predictions
)

test_f1 = f1_score(
    y_test_binary,
    test_predictions
)

test_roc_auc = roc_auc_score(
    y_test_binary,
    test_predictions
)

test_conf_matrix = confusion_matrix(
    y_test_binary,
    test_predictions
)

print("\n========== Test Dataset Results ==========")

print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1 Score : {test_f1:.4f}")
print(f"ROC-AUC  : {test_roc_auc:.4f}")

print("\nTest Confusion Matrix:")
print(test_conf_matrix)

with open("results/metrics/test_metrics.txt", "w") as file: #for saving the results
    file.write("Autoencoder Test Evaluation\n")
    file.write("=" * 35 + "\n\n")

    file.write(f"Threshold : {threshold_mean3sigma:.8f}\n")
    file.write(f"Accuracy  : {test_accuracy:.4f}\n")
    file.write(f"Precision : {test_precision:.4f}\n")
    file.write(f"Recall    : {test_recall:.4f}\n")
    file.write(f"F1 Score  : {test_f1:.4f}\n")
    file.write(f"ROC-AUC   : {test_roc_auc:.4f}\n")

print("Test metrics saved successfully!")

#....................................Ploting reconstruction error historgram.........................................
plt.figure(figsize=(10, 6))

plt.hist(
    test_errors,
    bins=100,
    alpha=0.7,
    label="Reconstruction Errors"
)

plt.axvline(
    threshold_mean3sigma,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Mean + 3σ Threshold"
)

plt.title("Test Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Samples")

plt.legend()
plt.grid(True)

plt.savefig("results/plots/reconstruction_error_histogram.png")

plt.show()

print("\nReconstruction error histogram saved successfully!")