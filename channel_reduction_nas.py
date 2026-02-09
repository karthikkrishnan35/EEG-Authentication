import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Make results reproducible
np.random.seed(42)
tf.random.set_seed(42)

# Create directory for results
os.makedirs('results', exist_ok=True)

print("Loading NPZ data...")
try:
    data_npz = np.load('data/preprocessed_data_transposed.npz')
    print(f"Available arrays in NPZ file: {data_npz.files}")
    
    # Print information about each array to help identify which one contains the EEG data
    for array_name in data_npz.files:
        print(f"\nArray '{array_name}':")
        print(f"  Shape: {data_npz[array_name].shape}")
        print(f"  Type: {data_npz[array_name].dtype}")
        
    # Determine which array has the EEG data (3D shape)
    eeg_key = None
    label_key = None
    
    for key in data_npz.files:
        if len(data_npz[key].shape) == 3:
            eeg_key = key
            print(f"\nFound likely EEG data in array '{eeg_key}' with shape {data_npz[eeg_key].shape}")
        elif len(data_npz[key].shape) == 1 or (len(data_npz[key].shape) == 2 and data_npz[key].shape[1] == 1):
            label_key = key
            print(f"Found likely label data in array '{label_key}' with shape {data_npz[label_key].shape}")
    
    # If we couldn't identify the EEG data array, ask user for input
    if eeg_key is None:
        print("\nCould not automatically identify EEG data array.")
        print("Please enter the name of the array containing EEG data:")
        print(f"Available options: {data_npz.files}")
        exit(1)
        
    # Load the data
    eeg_data = data_npz[eeg_key]
    print(f"\nLoaded EEG data from '{eeg_key}' with shape {eeg_data.shape}")
    
    # Get or create labels
    if label_key:
        labels = data_npz[label_key]
        print(f"Loaded labels from '{label_key}' with shape {labels.shape}")
        
        # Convert to binary if needed
        if len(np.unique(labels)) > 2:
            print("Converting multi-class labels to binary (first class is authorized, others are imposters)")
            binary_labels = np.zeros_like(labels)
            authorized_id = np.unique(labels)[0]  # First unique label is authorized
            binary_labels[labels == authorized_id] = 1
            print(f"Created binary labels: {np.sum(binary_labels)} authorized, {len(binary_labels)-np.sum(binary_labels)} imposters")
        else:
            binary_labels = labels
    else:
        # Create synthetic binary labels (20% authorized, 80% imposters)
        print("No labels found. Creating synthetic binary labels (20% authorized, 80% imposters)")
        n_samples = eeg_data.shape[0]
        n_auth = int(n_samples * 0.2)
        binary_labels = np.zeros(n_samples)
        auth_indices = np.random.choice(n_samples, n_auth, replace=False)
        binary_labels[auth_indices] = 1
        print(f"Created binary labels: {np.sum(binary_labels)} authorized, {n_samples-np.sum(binary_labels)} imposters")
        
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Make sure data has the right shape (samples, channels, timepoints)
n_samples, n_channels, n_timepoints = eeg_data.shape
print(f"Data dimensions: {n_samples} samples, {n_channels} channels, {n_timepoints} timepoints")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    eeg_data, binary_labels, test_size=0.2, stratify=binary_labels, random_state=42
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Define channel selection model with NAS
def create_channel_selection_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Channel attention mechanism
    #Global Average Pooling to get a summary of each channel’s contribution.
    channel_avg = layers.GlobalAveragePooling1D()(inputs)
    #Attention mechanism (Dense layer with sigmoid activation) to learn channel importance.
    channel_weights = layers.Dense(input_shape[0], activation='sigmoid')(channel_avg)
    channel_weights = layers.Reshape((input_shape[0], 1))(channel_weights)
    
    # Apply channel weights
    weighted_channels = layers.Multiply()([inputs, channel_weights])
    
    # Feature extraction
    #Temporal feature extraction using Conv1D layers to analyze EEG signal patterns.
    x = layers.Conv1D(64, kernel_size=10, activation='relu')(weighted_channels)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    channel_selector = Model(inputs, channel_weights)
    
    return model, channel_selector

print("\nCreating channel selection model...")
input_shape = (n_channels, n_timepoints)
model, channel_selector = create_channel_selection_model(input_shape)
model.summary()

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Compile and train model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate model
print("\nEvaluating model...")
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Extract channel importance
#The importance scores indicate how much each channel contributes to distinguishing users.
print("\nExtracting channel importance...")
channel_importance = channel_selector.predict(X_test)
avg_importance = np.mean(channel_importance, axis=0).flatten()

# Plot channel importance
plt.figure(figsize=(12, 6))
plt.bar(range(n_channels), avg_importance)
plt.xlabel('Channel Index')
plt.ylabel('Importance Weight')
plt.title('Channel Importance from NAS')
plt.tight_layout()
plt.savefig('results/channel_importance.png')
print("Saved channel importance plot to 'results/channel_importance.png'")

# Select top channels
num_channels_to_keep = 3  # Adjust based on your needs
top_channel_indices = np.argsort(avg_importance)[-num_channels_to_keep:] #Highest importance values correspond to the most relevant EEG channels.
print(f"\nTop {num_channels_to_keep} channels: {sorted(top_channel_indices)}")

# Create reduced dataset
reduced_data = eeg_data[:, top_channel_indices, :]
print(f"Reduced data shape: {reduced_data.shape}")

# Save the reduced data
os.makedirs('data/processed', exist_ok=True)
np.savez('data/processed/reduced_channel_data.npz', 
         eeg_data=reduced_data, 
         channel_indices=top_channel_indices,
         labels=binary_labels)

print("\nChannel reduction complete!")
print("Reduced data saved to 'data/processed/reduced_channel_data.npz'")