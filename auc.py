import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("models/eeg_authentication_model.h5")

# Load extracted test data
data = np.load("data/features/extracted_features.npz")
X = data["features"]
y = data["labels"]

# Reshape X if needed
X = X.reshape(X.shape[0], X.shape[1], 1)  # Assuming original shape is (samples, time_steps)

# Ensure X_test and y_test are available
from sklearn.model_selection import train_test_split

train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, stratify=y_temp, random_state=42)

print(f"Model and data loaded successfully. Test set shape: {X_test.shape}")
