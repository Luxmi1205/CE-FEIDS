import joblib
import tensorflow as tf
import numpy as np
import random

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

# Number of input features
input_dim = X_train.shape[1]
print(f"Number of input features: {input_dim}")

# Random Seed
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Defining input layer
input_layer = Input(shape=(input_dim,))

# Model Configuration
ENCODER_UNITS_1 = 32
ENCODER_UNITS_2 = 16
LATENT_DIM = 8

OUTPUT_ACTIVATION = "sigmoid"
HIDDEN_ACTIVATION = "relu"

LEARNING_RATE = 0.001

EPOCHS = 100
BATCH_SIZE = 256

# Encoder
x = Dense(ENCODER_UNITS_1, activation=HIDDEN_ACTIVATION)(input_layer)
x = Dense(ENCODER_UNITS_2, activation=HIDDEN_ACTIVATION)(x)
latent = Dense(LATENT_DIM, activation=HIDDEN_ACTIVATION, name="latent_space")(x)

# Decoder
x = Dense(ENCODER_UNITS_2, activation=HIDDEN_ACTIVATION)(latent)
x = Dense(ENCODER_UNITS_1, activation=HIDDEN_ACTIVATION)(x)
output_layer = Dense(input_dim, activation=OUTPUT_ACTIVATION)(x)

# Creating Autoencoder model
autoencoder = Model(
    inputs=input_layer,
    outputs=output_layer,
    name="CE_FEIDS_Autoencoder"
)

# Compiling Autoencoder
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse"
)

print("\nAutoencoder compiled successfully!")
print(f"Optimizer: Adam")
print(f"Learning Rate: {LEARNING_RATE}")
print("Loss Function: Mean Squared Error (MSE)\n")

# This configure Early Stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

print("EarlyStopping callback configured successfully!")

# Configure Model Checkpoint
model_checkpoint = ModelCheckpoint(
    filepath="models/autoencoder/best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

print("ModelCheckpoint callback configured successfully!")

# This display model summary
autoencoder.summary()

# Training the Autoencoder
history = autoencoder.fit(
    X_train,
    X_train,
    validation_data=(X_validation, X_validation),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint],
    shuffle=True,
    verbose=1
)

print("\nAutoencoder training completed successfully!")

# Saving training history
joblib.dump(
    history.history,
    "models/autoencoder/training_history.pkl"
)

print("Training history saved successfully!")