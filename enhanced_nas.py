import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy import signal

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories for results
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

def load_preprocessed_data():
    """Load and combine preprocessed data from all subjects"""
    preprocessed_dir = 'data/preprocessed'
    
    # Get a list of all preprocessed files
    npz_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('_preprocessed.npz')]
    
    if not npz_files:
        raise FileNotFoundError(f"No preprocessed data found in {preprocessed_dir}")
    
    # Prepare lists to store data
    all_eeg_data = []
    all_labels = []
    channel_names = None
    authorized_subject = 'S001'  # Change this if needed
    
    print(f"Loading preprocessed data files...")
    for npz_file in npz_files:
        subject_id = npz_file.split('_')[0]
        npz_path = os.path.join(preprocessed_dir, npz_file)
        
        # Load data
        data = np.load(npz_path)
        eeg_data = data['eeg_data']
        
        if channel_names is None:
            channel_names = data['channel_names']
        elif len(channel_names) != eeg_data.shape[0]:
            print(f"Warning: Channel count mismatch in {npz_file}. Skipping.")
            continue
        
        # Create segments
        segment_size = 2 * int(data['sampling_rate'])  # 2 seconds
        step_size = segment_size // 2  # 50% overlap
        
        n_segments = (eeg_data.shape[1] - segment_size) // step_size + 1
        segments = np.zeros((n_segments, eeg_data.shape[0], segment_size))
        
        for i in range(n_segments):
            start = i * step_size
            end = start + segment_size
            segments[i] = eeg_data[:, start:end]
        
        # Create labels (1 for authorized subject, 0 for others)
        labels = np.ones(n_segments) if subject_id == authorized_subject else np.zeros(n_segments)
        
        # Add to our dataset
        all_eeg_data.append(segments)
        all_labels.append(labels)
        
        print(f"Added {n_segments} segments from {subject_id}")
    
    # Combine data
    eeg_data = np.vstack(all_eeg_data)
    labels = np.hstack(all_labels)
    
    print(f"Final dataset: {eeg_data.shape[0]} segments, {eeg_data.shape[1]} channels, {eeg_data.shape[2]} timepoints")
    print(f"Authorized samples: {np.sum(labels)}, Unauthorized: {len(labels) - np.sum(labels)}")
    
    return eeg_data, labels, channel_names

def compute_frequency_features(eeg_data, sampling_rate=128):
    """Compute frequency domain features for each segment"""
    n_samples, n_channels, n_timepoints = eeg_data.shape
    
    # Calculate power spectral density features
    # We'll create band power features for delta, theta, alpha, beta, gamma
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Initialize array to hold frequency features
    freq_features = np.zeros((n_samples, n_channels, len(bands)))
    
    for i in range(n_samples):
        for j in range(n_channels):
            # Compute power spectral density
            f, psd = signal.welch(eeg_data[i, j], fs=sampling_rate, nperseg=min(256, n_timepoints))
            
            # Extract band powers
            for k, (band_name, (low_freq, high_freq)) in enumerate(bands.items()):
                # Find frequency band indices
                idx_band = np.logical_and(f >= low_freq, f <= high_freq)
                # Calculate power in band
                if np.any(idx_band):
                    freq_features[i, j, k] = np.mean(psd[idx_band])
                else:
                    freq_features[i, j, k] = 0
    
    return freq_features

def create_advanced_channel_attention_model(time_input_shape, freq_input_shape=None):
    """Create an advanced model with hierarchical attention mechanisms"""
    
    # Time-domain input branch
    time_inputs = layers.Input(shape=time_input_shape, name='time_input')
    
    # Multi-scale feature extraction
    # Path 1: Global features
    global_features = layers.Lambda(lambda x: tf.reduce_mean(x, axis=2, keepdims=True))(time_inputs)
    
    # Path 2: Local temporal features with different kernel sizes
    conv1 = layers.Conv1D(32, kernel_size=3, padding='same', activation=None)(time_inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('elu')(conv1)
    
    conv2 = layers.Conv1D(32, kernel_size=7, padding='same', activation=None)(time_inputs)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('elu')(conv2)
    
    conv3 = layers.Conv1D(32, kernel_size=15, padding='same', activation=None)(time_inputs)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('elu')(conv3)
    
    # Combine multi-scale features
    multi_scale_features = layers.Concatenate()([conv1, conv2, conv3])
    
    # Add recurrent layer to capture temporal dependencies
    rnn_features = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(multi_scale_features)
    
    # Add frequency domain input if provided
    if freq_input_shape is not None:
        freq_inputs = layers.Input(shape=freq_input_shape, name='freq_input')
        # Process frequency features
        freq_conv = layers.Conv1D(16, kernel_size=3, padding='same')(freq_inputs)
        freq_conv = layers.BatchNormalization()(freq_conv)
        freq_conv = layers.Activation('elu')(freq_conv)
        
        # Combine with time domain features (need to reshape for compatibility)
        # Assume freq_input_shape is (channels, bands)
        freq_reshaped = layers.Reshape((freq_input_shape[0], freq_input_shape[1], 1))(freq_inputs)
        freq_conv = layers.Conv2D(1, kernel_size=(1, freq_input_shape[1]), padding='valid')(freq_reshaped)
        freq_features = layers.Reshape((freq_input_shape[0], 1))(freq_conv)
        
        # Concatenate with global features
        combined_global = layers.Concatenate(axis=-1)([global_features, freq_features])
    else:
        combined_global = global_features
        freq_inputs = None
    
    # Hierarchical attention mechanism
    
    # Level 1: Channel attention based on global features
    channel_attn = layers.Dense(64, activation='elu', 
                              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(combined_global)
    channel_attn = layers.Dense(32, activation='elu')(channel_attn)
    channel_attn = layers.Dense(1, activation='sigmoid')(channel_attn)
    
    # Apply channel attention weights
    weighted_channels = layers.Multiply()([time_inputs, channel_attn])
    
    # Level 2: Temporal attention using both RNN features and original weighted channels
    temporal_context = layers.Concatenate()([weighted_channels, rnn_features])
    
    # Reduce the channel dimension to make computation more efficient
    channel_pooled = layers.AveragePooling1D(pool_size=4)(temporal_context)
    
    # Final feature extraction
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='elu',
                    kernel_regularizer=regularizers.l2(1e-4))(channel_pooled)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Global pooling and fully connected layers
    x = layers.GlobalAveragePooling1D()(x)
    
    # Add dropout for regularization
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='elu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create the model
    if freq_input_shape is not None:
        model = Model([time_inputs, freq_inputs], outputs)
        channel_importance_model = Model([time_inputs, freq_inputs], channel_attn)
    else:
        model = Model(time_inputs, outputs)
        channel_importance_model = Model(time_inputs, channel_attn)
    
    return model, channel_importance_model

def visualize_channel_importance(importance_weights, channel_names, top_n=3):
    """Visualize channel importance with various plots"""
    
    # Sort channels by importance
    sorted_indices = np.argsort(importance_weights)
    
    # Create directory for channel visualizations
    channel_viz_dir = os.path.join(results_dir, 'channel_visualization')
    os.makedirs(channel_viz_dir, exist_ok=True)
    
    # 1. Bar plot of channel importance
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(importance_weights)), importance_weights, color='skyblue')
    plt.xlabel('Channel Index')
    plt.ylabel('Importance Weight')
    plt.title('EEG Channel Importance from Advanced Attention Model')
    
    # Highlight top N channels
    top_indices = sorted_indices[-top_n:]
    for i in top_indices:
        plt.bar(i, importance_weights[i], color='red')
        plt.text(i, importance_weights[i], f"{i} ({channel_names[i] if i < len(channel_names) else 'Ch'+str(i)})", 
                 ha='center', va='bottom', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(channel_viz_dir, 'channel_importance_bar.png'))
    
    # 2. Create a head map visualization
    try:
        from mne.viz import plot_topomap
        from mne.channels import make_standard_montage
        import mne
        
        # Try to create a standard 10-20 montage
        montage = make_standard_montage('standard_1020')
        
        # Create info object
        info = mne.create_info(
            ch_names=list(channel_names),
            sfreq=250,
            ch_types=['eeg'] * len(channel_names)
        )
        
        # Set montage
        info.set_montage(montage)
        
        # Create a topographic map of channel importance
        plt.figure(figsize=(10, 8))
        im, cn = plot_topomap(importance_weights, info, names=channel_names, 
                              show_names=True, contours=0, vmin=0, vmax=np.max(importance_weights))
        plt.colorbar(im)
        plt.title('Channel Importance Topographic Map')
        plt.savefig(os.path.join(channel_viz_dir, 'channel_importance_topomap.png'))
        
    except Exception as e:
        print(f"Skipping topographic map due to error: {e}")
    
    # 3. Simplified EEG montage visualization (if MNE fails)
    plt.figure(figsize=(10, 10))
    
    # Create a simplified 10-20 system layout
    positions = {
        'Fp1': [-0.2, 0.9], 'Fp2': [0.2, 0.9],
        'F7': [-0.7, 0.7], 'F3': [-0.4, 0.6], 'Fz': [0, 0.6], 'F4': [0.4, 0.6], 'F8': [0.7, 0.7],
        'T7': [-0.9, 0], 'C3': [-0.4, 0], 'Cz': [0, 0], 'C4': [0.4, 0], 'T8': [0.9, 0],
        'P7': [-0.7, -0.7], 'P3': [-0.4, -0.6], 'Pz': [0, -0.6], 'P4': [0.4, -0.6], 'P8': [0.7, -0.7],
        'O1': [-0.2, -0.9], 'O2': [0.2, -0.9]
    }
    
    # Plot a head outline
    circle = plt.Circle((0, 0), 1, fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    
    # Add a nose
    plt.plot([0, 0], [1, 1.1], 'k', linewidth=2)
    
    # Add ears
    plt.plot([-1, -1.1], [0, 0], 'k', linewidth=2)
    plt.plot([1, 1.1], [0, 0], 'k', linewidth=2)
    
    # Plot each channel
    channel_dict = {ch: i for i, ch in enumerate(channel_names)}
    
    for ch_name, pos in positions.items():
        if ch_name in channel_dict:
            ch_idx = channel_dict[ch_name]
            importance = importance_weights[ch_idx]
            is_top = ch_idx in top_indices
            
            # Normalize size and color by importance
            size = 100 + 900 * importance / np.max(importance_weights)
            color = 'red' if is_top else plt.cm.viridis(importance / np.max(importance_weights))
            
            plt.scatter(pos[0], pos[1], s=size, c=[color], alpha=0.7)
            plt.text(pos[0], pos[1], ch_name, ha='center', va='center', 
                     fontsize=8 + 4 * importance / np.max(importance_weights),
                     weight='bold' if is_top else 'normal')
    
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.axis('off')
    plt.title('Channel Importance on EEG Montage\n(Red = Top 3)', fontsize=14)
    plt.savefig(os.path.join(channel_viz_dir, 'channel_importance_montage.png'))
    
    # 4. Heatmap of channel importance
    plt.figure(figsize=(12, 10))
    
    # Create a 2D grid for a heatmap (6x6 grid should fit all channels)
    grid_size = 6
    grid = np.zeros((grid_size, grid_size))
    
    # Map channel positions to grid
    grid_positions = {
        'Fp1': (0, 1), 'Fp2': (0, 4),
        'F7': (1, 0), 'F3': (1, 1), 'Fz': (1, 2), 'F4': (1, 3), 'F8': (1, 4),
        'T7': (2, 0), 'C3': (2, 1), 'Cz': (2, 2), 'C4': (2, 3), 'T8': (2, 4),
        'P7': (3, 0), 'P3': (3, 1), 'Pz': (3, 2), 'P4': (3, 3), 'P8': (3, 4),
        'O1': (4, 1), 'O2': (4, 3)
    }
    
    # Place importance values in grid
    for ch_name, pos in grid_positions.items():
        if ch_name in channel_dict:
            ch_idx = channel_dict[ch_name]
            grid[pos] = importance_weights[ch_idx]
    
    # Apply Gaussian smoothing for better visualization
    smoothed_grid = gaussian_filter(grid, sigma=0.8)
    
    # Plot heatmap
    sns.heatmap(smoothed_grid, cmap='viridis', cbar_kws={'label': 'Importance'})
    
    # Add channel labels
    for ch_name, pos in grid_positions.items():
        if ch_name in channel_dict:
            ch_idx = channel_dict[ch_name]
            weight = "%.3f" % importance_weights[ch_idx]
            color = 'white' if importance_weights[ch_idx] > np.max(importance_weights) / 2 else 'black'
            text = f"{ch_name}\n{weight}"
            if ch_idx in top_indices:
                text = f"★{ch_name}★\n{weight}"
                plt.text(pos[1] + 0.5, pos[0] + 0.5, text, ha='center', va='center', 
                         color='red', fontweight='bold', fontsize=10)
            else:
                plt.text(pos[1] + 0.5, pos[0] + 0.5, text, ha='center', va='center', color=color)
    
    plt.title('EEG Channel Importance Heatmap', fontsize=16)
    plt.savefig(os.path.join(channel_viz_dir, 'channel_importance_heatmap.png'))
    plt.close()
    
    return top_indices

def main():
    # Load preprocessed data
    print("Loading preprocessed EEG data...")
    eeg_data, labels, channel_names = load_preprocessed_data()
    
    # Check if channel_names is not None and has the right length
    if channel_names is None or len(channel_names) != eeg_data.shape[1]:
        print("Warning: Channel names missing or incorrect. Using generic channel names.")
        channel_names = [f"Channel_{i}" for i in range(eeg_data.shape[1])]
    
    # Reshape if needed (should be [samples, channels, timepoints])
    n_samples, n_channels, n_timepoints = eeg_data.shape
    
    # Compute frequency domain features
    print("Computing frequency domain features...")
    freq_features = compute_frequency_features(eeg_data)
    print(f"Frequency features shape: {freq_features.shape}")
    
    # Split data
    print("Splitting data into train/validation/test sets...")
    # First, create a development set and a hold-out test set
    X_dev, X_test, y_dev, y_test = train_test_split(
        eeg_data, labels, test_size=0.15, stratify=labels, random_state=42
    )
    
    # Split development set into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.2, stratify=y_dev, random_state=42
    )
    
    # Also split frequency features
    freq_dev, freq_test, _, _ = train_test_split(
        freq_features, labels, test_size=0.15, stratify=labels, random_state=42
    )
    
    freq_train, freq_val, _, _ = train_test_split(
        freq_dev, y_dev, test_size=0.2, stratify=y_dev, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create the advanced model with channel attention
    print("\nCreating advanced model with hierarchical attention...")
    model, channel_importance_model = create_advanced_channel_attention_model(
        (n_channels, n_timepoints),
        (n_channels, freq_features.shape[2])
    )
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Print model summary
    model.summary()
    
    # Prepare callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        [X_train, freq_train], y_train,
        validation_data=([X_val, freq_val], y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        class_weight={0: 1.0, 1: len(y_train) / max(1, sum(y_train))},  # Balance classes
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_results = model.evaluate([X_test, freq_test], y_test, verbose=1)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    print(f"Test AUC: {test_results[2]:.4f}")
    
    # Extract channel importance
    print("\nExtracting channel importance...")
    # Use validation set to compute importance
    importance_batch_size = min(100, X_val.shape[0])
    channel_weights = channel_importance_model.predict(
        [X_val[:importance_batch_size], freq_val[:importance_batch_size]]
    )
    
    # Average importance weights across samples
    avg_importance = np.mean(channel_weights, axis=0).flatten()
    
    # Normalize for visualization
    normalized_importance = avg_importance / np.max(avg_importance)
    
    # Visualize channel importance and get top channels
    print("\nVisualizing channel importance...")
    top_channel_indices = visualize_channel_importance(normalized_importance, channel_names)
    
    # Create reduced dataset with only top 3 channels
    print("\nCreating reduced dataset with top 3 channels...")
    reduced_data = eeg_data[:, top_channel_indices, :]
    
    # Get original channel names/numbers
    top_channels = [channel_names[i] if i < len(channel_names) else f"Ch{i}" for i in top_channel_indices]
    print(f"Top 3 channels selected: {top_channels} (indices: {top_channel_indices})")
    
    # Save reduced dataset
    reduced_data_path = os.path.join(results_dir, 'reduced_3channel_data.npz')
    np.savez(reduced_data_path, 
             eeg_data=reduced_data, 
             labels=labels,
             channel_indices=top_channel_indices,
             channel_names=[channel_names[i] for i in top_channel_indices] if len(channel_names) > max(top_channel_indices) else None)
    print(f"Reduced dataset saved to {reduced_data_path}")

    # Visualize training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))

    # Verify the selected channels by creating a small model trained only on these channels
    print("\nVerifying selected channels with a simple test model...")
    
    # Create and train a simple model using only the selected channels
    X_train_reduced = X_train[:, top_channel_indices, :]
    X_val_reduced = X_val[:, top_channel_indices, :]
    X_test_reduced = X_test[:, top_channel_indices, :]
    
    # Simple CNN model
    verification_model = tf.keras.Sequential([
        layers.Conv1D(32, kernel_size=8, activation='relu', input_shape=(3, n_timepoints)),
        layers.MaxPooling1D(4),
        layers.Conv1D(64, kernel_size=4, activation='relu'),
        layers.MaxPooling1D(4),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    verification_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Train the verification model
    verification_model.fit(
        X_train_reduced, y_train,
        validation_data=(X_val_reduced, y_val),
        epochs=15,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate verification model
    verification_results = verification_model.evaluate(X_test_reduced, y_test, verbose=1)
    print(f"Verification model with only 3 channels:")
    print(f"Test loss: {verification_results[0]:.4f}")
    print(f"Test accuracy: {verification_results[1]:.4f}")
    print(f"Test AUC: {verification_results[2]:.4f}")
    
    print("\nChannel reduction complete!")
    print(f"Top 3 channels selected: {top_channels}")

if __name__ == "__main__":
    main()