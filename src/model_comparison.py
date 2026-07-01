import pandas as pd

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

centralized_metrics = load_metrics(
    "results/metrics/test_metrics.txt"
)

federated_metrics = load_metrics(
    "results/metrics/federated_metrics.txt"
)

comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],

    "Centralized": [
        centralized_metrics["Accuracy"],
        centralized_metrics["Precision"],
        centralized_metrics["Recall"],
        centralized_metrics["F1 Score"],
        centralized_metrics["ROC-AUC"]
    ],

    "Federated": [
        federated_metrics["Accuracy"],
        federated_metrics["Precision"],
        federated_metrics["Recall"],
        federated_metrics["F1 Score"],
        federated_metrics["ROC-AUC"]
    ]
})

comparison.to_csv(
    "results/metrics/model_comparison.csv",
    index=False
)

print(comparison)

print("\nModel comparison table saved successfully!")