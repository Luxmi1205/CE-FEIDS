import joblib
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load benign training data
X_train = joblib.load(
    "data/processed/X_train_benign_scaled.pkl"
)

print("Training dataset loaded successfully!")
print(f"Training Data Shape: {X_train.shape}")

# Spliting the data into 4 federated clients as to simulate four federated clients by partitioning the benign training dataset
clients = np.array_split(X_train, 4)

print("\nTraining data split into 4 clients successfully!")

for i, client_data in enumerate(clients, start=1):
    print(f"Client {i} Shape: {client_data.shape}")

# Defining model constants
ENCODER_UNITS_1 = 32
ENCODER_UNITS_2 = 16
LATENT_DIM = 8

HIDDEN_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "sigmoid"
LEARNING_RATE = 0.001

LOCAL_EPOCHS = 3
BATCH_SIZE = 256
COMMUNICATION_ROUNDS = 10


# Model function
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))

    # Encoder
    x = Dense(ENCODER_UNITS_1, activation=HIDDEN_ACTIVATION)(input_layer)
    x = Dense(ENCODER_UNITS_2, activation=HIDDEN_ACTIVATION)(x)
    latent = Dense(LATENT_DIM, activation=HIDDEN_ACTIVATION)(x)

    # Decoder
    x = Dense(ENCODER_UNITS_2, activation=HIDDEN_ACTIVATION)(latent)
    x = Dense(ENCODER_UNITS_1, activation=HIDDEN_ACTIVATION)(x)
    output_layer = Dense(input_dim, activation=OUTPUT_ACTIVATION)(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mse"
    )

    return model


def train_local_client(model, client_data, client_id):
    print(f"\nTraining Client {client_id}...")

    history = model.fit(
        client_data,
        client_data,
        epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    print(f"\nClient {client_id} training completed successfully!")

    return model, history


def federated_average(local_models):
    #Average the weights of all local models to create a global model.

    global_model = build_autoencoder(X_train.shape[1])

    averaged_weights = []

    # Get weights from all local models
    local_weights = [model.get_weights() for model in local_models]

    # Average each layer
    for layer_weights in zip(*local_weights):
        averaged_weights.append(
            np.mean(layer_weights, axis=0)
        )

    # Set averaged weights
    global_model.set_weights(averaged_weights)

    print("\nFederated Averaging completed successfully!")

    return global_model


# Main function for keeping code clean and reusable def main():
def main():
    input_dim = X_train.shape[1]

    # Initial global model
    global_model = build_autoencoder(input_dim)

    print("Initial Global Model created successfully!")

    # Communication rounds
    for round_num in range(COMMUNICATION_ROUNDS):

        print(f"\n{'=' * 60}")
        print(f"Communication Round {round_num + 1}")
        print(f"{'=' * 60}")

        local_models = []

        # Train each client
        for i, client_data in enumerate(clients, start=1):

            print(f"\n{'=' * 50}")
            print(f"Federated Client {i}")
            print(f"{'=' * 50}")

            model = build_autoencoder(input_dim)

            # Send global weights to client
            model.set_weights(
                global_model.get_weights()
            )

            trained_model, history = train_local_client(
                model,
                client_data,
                client_id=i
            )

            local_models.append(trained_model)

        print("\nCreating Global Model...")

        global_model = federated_average(local_models)

        print("\nGlobal Model updated successfully!")

    # Save only the final global model
    os.makedirs(
        "models/federated",
        exist_ok=True
    )

    global_model.save(
        "models/federated/global_autoencoder.keras"
    )

    print("\nFinal global federated model saved successfully!")


# Calling the main function 
if __name__ == "__main__":
    main()