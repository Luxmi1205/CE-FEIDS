import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load processed datasets
X_train = joblib.load("data/processed/X_train_benign_scaled.pkl")
X_validation = joblib.load("data/processed/X_validation_scaled.pkl")
X_test = joblib.load("data/processed/X_test_scaled.pkl")

y_validation = joblib.load("data/processed/y_validation.pkl")
y_test = joblib.load("data/processed/y_test.pkl")

print("Processed datasets loaded successfully!\n")

print(f"Training Data Shape: {X_train.shape}")
print(f"Validation Data Shape: {X_validation.shape}")
print(f"Test Data Shape: {X_test.shape}")

# Defining input layer
input_layer = Input(shape=(44,))

# Encoder
encoder = Dense(32, activation="relu")(input_layer)
encoder = Dense(16, activation="relu")(encoder)
latent = Dense(8, activation="relu", name="latent_space")(encoder)

# Decoder
decoder = Dense(16, activation="relu")(latent)
decoder = Dense(32, activation="relu")(decoder)
output_layer = Dense(44, activation="sigmoid")(decoder)

# Creating Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# This display model summary
autoencoder.summary()