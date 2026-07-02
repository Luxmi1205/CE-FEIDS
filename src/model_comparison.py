import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def load_metrics(file_path):
    # this loads evaluation metrics from the text file
    metrics = {}
    with open(file_path, "r") as file:
        for line in file:
            if ":" in line:
                key, value = line.split(":", 1)
                try:
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    continue
    return metrics

centralized_metrics = load_metrics("results/metrics/test_metrics.txt")

federated_metrics = load_metrics("results/metrics/federated_metrics.txt")

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"],

    "Centralized": [
        centralized_metrics["Accuracy"],
        centralized_metrics["Precision"],
        centralized_metrics["Recall"],
        centralized_metrics["F1 Score"],
        centralized_metrics["ROC-AUC"]],

    "Federated": [
        federated_metrics["Accuracy"],
        federated_metrics["Precision"],
        federated_metrics["Recall"],
        federated_metrics["F1 Score"],
        federated_metrics["ROC-AUC"]]
}
)
comparison.to_csv(
    "results/metrics/model_comparison.csv",
    index=False)
print(comparison)

print("\nModel comparison table saved successfully!")

# Creating a comparison bar chart for a given evaluation metric
def create_metric_plot(metric_name, filename):
    plt.figure(figsize=(6, 5))

    models = ["Centralized", "Federated"]

    values = [
        centralized_metrics[metric_name],
        federated_metrics[metric_name]]

    plt.bar(models, values)

    plt.title(f"{metric_name} Comparison")

    plt.ylabel(metric_name)

    #plt.ylim(0.85, 1.0)
    plt.ylim(0, 1.0)

    plt.savefig(
        f"results/plots/{filename}",
        dpi=300,
        bbox_inches="tight")
    plt.close()

    print(f"{metric_name} comparison plot saved successfully!")

# Creating comparison plots
os.makedirs(
    "results/plots",
    exist_ok=True)

create_metric_plot(
    "Accuracy",
    "accuracy_comparison.png")

create_metric_plot(
    "Precision",
    "precision_comparison.png")

create_metric_plot(
    "Recall",
    "recall_comparison.png")

create_metric_plot(
    "F1 Score",
    "f1_comparison.png")

create_metric_plot(
    "ROC-AUC",
    "roc_auc_comparison.png")

# Creating overall comparison chart
plt.figure(figsize=(10, 6))
metrics = comparison["Metric"]

x = np.arange(len(metrics))
width = 0.35

plt.bar(
    x - width / 2,
    comparison["Centralized"],
    width,
    label="Centralized")

plt.bar(
    x + width / 2,
    comparison["Federated"],
    width,
    label="Federated")

plt.xticks(x, metrics)

plt.ylabel("Score")

plt.title("Centralized vs Federated Model Performance")

plt.ylim(0, 1.05)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/plots/overall_metrics_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Overall metrics comparison plot saved successfully")

# Creating final evaluation summary
with open("results/metrics/evaluation_summary.txt","w") as file:
    file.write("CE-FEIDS Evaluation Summary\n")
    file.write("=" * 45 + "\n\n")

    file.write("Centralized Autoencoder\n")
    file.write("-" * 30 + "\n")

    for metric, value in centralized_metrics.items():
        file.write(f"{metric:<12}: {value:.4f}\n")

    file.write("\nFederated Autoencoder\n")
    file.write("-" * 30 + "\n")

    for metric, value in federated_metrics.items():
        file.write(f"{metric:<12}: {value:.4f}\n")

    file.write("\nObservations\n")
    file.write("-" * 30 + "\n")

    file.write("Federated learning achieved performance close to the centralized model while securing decentralized training.\n")

    file.write("Precision remained extremely high & indicating very few false alarms.\n")

    file.write("Recall is slightly decreased due to the conservative anomaly thresholding\n")

    file.write("ROC-AUC remaines competitive & demonstrating strong differentiation capability.\n")
print("Evaluation summary generated successfully")