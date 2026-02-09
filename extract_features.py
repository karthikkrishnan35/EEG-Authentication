import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

# Load extracted features
data = np.load("data/features/extracted_features.npz")
X = data["features"]  # EEG feature matrix
y = data["labels"]  # Labels

print(f"Loaded Features Shape: {X.shape}")
print(f"Loaded Labels Shape: {y.shape}")


# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)

# Split data ensuring no subject leakage
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, stratify=y_temp, random_state=42)

# Define model with dropout and L2 regularization
model = keras.Sequential([
    layers.Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), input_shape=(X.shape[1], X.shape[2])),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile model with lower learning rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Train model with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
