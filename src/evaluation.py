import joblib
import matplotlib.pyplot as plt

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