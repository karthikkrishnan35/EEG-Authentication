import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split  # Add this import

# Load the test data (features and labels)
data = np.load("data/features/extracted_features.npz")
X = data["features"]  # EEG feature matrix
y = data["labels"]  # Labels

# Reshape the data to match the model input shape (samples, time_steps, channels)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data (ensure that y_test is properly defined)
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, stratify=y_temp, random_state=42)

# Load the trained model
model = tf.keras.models.load_model("models/eeg_authentication_model.h5")

# Predict probabilities
y_probs = model.predict(X_test)

# Convert probabilities to binary predictions (threshold at 0.5)
y_pred = (y_probs > 0.5).astype(int)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)

# Print the evaluation metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-score: {f1 * 100:.2f}%")
print(f"AUC: {roc_auc:.4f}")
