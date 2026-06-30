import pandas as pd

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Centralized": [
        0.9141,
        0.9987,
        0.9133,
        0.9541,
        0.9312
    ],
    "Federated": [
        0.9016,
        0.9997,
        0.8995,
        0.9470,
        0.9446
    ]
})

comparison.to_csv(
    "results/metrics/model_comparison.csv",
    index=False
)

print(comparison)

print("\nModel comparison table saved successfully!")