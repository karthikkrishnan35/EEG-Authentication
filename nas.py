import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.ndimage import gaussian_filter

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

def create_nas_channel_attention_model(input_shape):
    """Create model with neural architecture search style channel attention"""
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Channel attention mechanism
    # First, get a global representation of each channel
    channel_avg = layers.GlobalAveragePooling1D()(inputs)
    channel_max = layers.GlobalMaxPooling1D()(inputs)
    
    # Process the pooled features
    avg_features = layers.Dense(input_shape[0] // 2, activation='relu')(channel_avg)
    max_features = layers.Dense(input_shape[0] // 2, activation='relu')(channel_max)
    
    # Combine the features
    channel_features = layers.Concatenate()([avg_features, max_features])
    
    # Generate attention weights
    channel_attention = layers.Dense(input_shape[0], activation='sigmoid')(channel_features)
    
    # Reshape to apply as channel weights
    channel_weights = layers.Reshape((input_shape[0], 1))(channel_attention)
    
    # Apply channel weights
    weighted_channels = layers.Multiply()([inputs, channel_weights])
    
    # Feature extraction using CNN
    x = layers.Conv1D(32, kernel_size=8, padding='same', activation='relu')(weighted_channels)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Conv1D(64, kernel_size=4, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Conv1D(128, kernel_size=2, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Fully connected layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create main model
    model = Model(inputs, outputs)
    
    # Create a separate model that outputs channel weights
    channel_importance_model = Model(inputs, channel_weights)
    
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
    plt.title('EEG Channel Importance from NAS-Attention Model')
    
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
    
    # Reshape if needed (should be [samples, channels, timepoints])
    n_samples, n_channels, n_timepoints = eeg_data.shape
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        eeg_data, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create the model with channel attention
    print("\nCreating model with NAS channel attention...")
    model, channel_importance_model = create_nas_channel_attention_model((n_channels, n_timepoints))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train the model
    print("\nTraining model...")
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Extract channel importance
    print("\nExtracting channel importance...")
    # Use a subset of data to compute importance
    importance_batch = X_test[:100]  
    channel_weights = channel_importance_model.predict(importance_batch)
    
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
    
    # Get original channel numbers
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
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))

    print("\nChannel reduction complete!")
    print(f"Top 3 channels selected: {top_channels}")

if __name__ == "__main__":
    main()