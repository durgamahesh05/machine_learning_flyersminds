"""
ANN Hidden Neuron Experiment
Experiment: Compare 5, 20 and 50 hidden neurons.
Dataset: Iris classification dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load classification dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Experiment settings
neuron_counts = [5, 20, 50]
results = []

for neurons in neuron_counts:
    print(f"\nTraining ANN with {neurons} hidden neurons...")

    # ANN: input -> one hidden layer -> output
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(neurons, activation="relu"),
        keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )

    # Predictions
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    results.append({
        "neurons": neurons,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "test_loss": test_loss,
        "final_train_loss": history.history["loss"][-1],
        "final_val_loss": history.history["val_loss"][-1]
    })

# Display results
print("\nRESULTS")
print("=" * 90)
print(f"{'Neurons':<10}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Test Loss':<12}")
print("-" * 90)

for result in results:
    print(
        f"{result['neurons']:<10}"
        f"{result['accuracy']:<12.4f}"
        f"{result['precision']:<12.4f}"
        f"{result['recall']:<12.4f}"
        f"{result['f1_score']:<12.4f}"
        f"{result['test_loss']:<12.4f}"
    )

# Plot accuracy comparison
neurons = [r["neurons"] for r in results]
accuracies = [r["accuracy"] for r in results]

plt.figure(figsize=(7, 5))
plt.plot(neurons, accuracies, marker="o")
plt.xlabel("Number of Hidden Neurons")
plt.ylabel("Test Accuracy")
plt.title("ANN Performance vs Hidden Neurons")
plt.xticks(neurons)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot test loss comparison
losses = [r["test_loss"] for r in results]

plt.figure(figsize=(7, 5))
plt.plot(neurons, losses, marker="o")
plt.xlabel("Number of Hidden Neurons")
plt.ylabel("Test Loss")
plt.title("ANN Test Loss vs Hidden Neurons")
plt.xticks(neurons)
plt.grid(True)
plt.tight_layout()
plt.show()
