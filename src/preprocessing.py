import pandas as pd
import os
import joblib

# Loading datasets
train_df = pd.read_csv("data/train.csv")
validation_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

print("Datasets loaded successfully!\n")
print(f"Train Shape: {train_df.shape}")
print(f"Validation Shape: {validation_df.shape}")
print(f"Test Shape: {test_df.shape}")


# Separating features and labels
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_validation = validation_df.drop(columns=["label"])
y_validation = validation_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

print("\nLabel column separated successfully!\n")

print(f"Training Features Shape: {X_train.shape}")
print(f"Training Labels Shape: {y_train.shape}")

print(f"Validation Features Shape: {X_validation.shape}")
print(f"Validation Labels Shape: {y_validation.shape}")

print(f"Test Features Shape: {X_test.shape}")
print(f"Test Labels Shape: {y_test.shape}")


# Removing constant features
constant_features = ["IRC", "Telnet"]

X_train = X_train.drop(columns=constant_features)
X_validation = X_validation.drop(columns=constant_features)
X_test = X_test.drop(columns=constant_features)

print("\nConstant features removed successfully!\n")
print(f"Training Features Shape: {X_train.shape}")
print(f"Validation Features Shape: {X_validation.shape}")
print(f"Test Features Shape: {X_test.shape}")


# Separating benign traffic
X_train_benign = X_train[y_train == "BenignTraffic"]

X_validation_benign = X_validation[y_validation == "BenignTraffic"]

X_test_benign = X_test[y_test == "BenignTraffic"]

print("\nBenign traffic separated successfully!\n")

print(f"Training Benign Samples: {X_train_benign.shape}")
print(f"Validation Benign Samples: {X_validation_benign.shape}")
print(f"Test Benign Samples: {X_test_benign.shape}")

from sklearn.preprocessing import MinMaxScaler

# Initializing MinMaxScaler
scaler = MinMaxScaler()

# Fit scaler only on benign training data
scaler.fit(X_train_benign)
print("\nMinMaxScaler fitted successfully on benign training data!")
print(f"Number of features learned by scaler: {scaler.n_features_in_}")

# Transforming datasets
X_train_benign_scaled = scaler.transform(X_train_benign)
X_validation_benign_scaled = scaler.transform(X_validation_benign)
X_test_benign_scaled = scaler.transform(X_test_benign)

X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

print("\nDatasets scaled successfully!\n")
print(f"Scaled Training Benign Shape: {X_train_benign_scaled.shape}")
print(f"Scaled Validation Shape: {X_validation_scaled.shape}")
print(f"Scaled Test Shape: {X_test_scaled.shape}")

# Creating models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Saving fitted scaler
scaler_path = "models/minmax_scaler.pkl"
joblib.dump(scaler, scaler_path)

print("\nScaler saved successfully!")
print(f"Scaler saved to: {scaler_path}")

# Creating processed data directory if it doesn't exist
processed_data_dir = "data/processed"
os.makedirs(processed_data_dir, exist_ok=True)

# This saves processed datasets
# .pkl preserves NumPy arrays and pandas objects without conversion
joblib.dump(X_train_benign_scaled, f"{processed_data_dir}/X_train_benign_scaled.pkl")
joblib.dump(X_validation_benign_scaled, f"{processed_data_dir}/X_validation_benign_scaled.pkl")
joblib.dump(X_test_benign_scaled, f"{processed_data_dir}/X_test_benign_scaled.pkl")

joblib.dump(X_validation_scaled, f"{processed_data_dir}/X_validation_scaled.pkl")
joblib.dump(X_test_scaled, f"{processed_data_dir}/X_test_scaled.pkl")

joblib.dump(y_validation, f"{processed_data_dir}/y_validation.pkl")
joblib.dump(y_test, f"{processed_data_dir}/y_test.pkl")

print("\nProcessed datasets saved successfully!")
print(f"Saved to: {processed_data_dir}")