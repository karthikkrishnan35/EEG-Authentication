import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

# Make results reproducible
np.random.seed(42)
tf.random.set_seed(42)

# Create directory for visualizations
VIZ_DIR = Path('results/visualizations')
VIZ_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the preprocessed data and reduced channel data"""
    print("Loading preprocessed data...")
    full_data = np.load('data/preprocessed_data_transposed.npz')
    
    # Find the EEG key
    eeg_key = None
    for key in full_data.files:
        if len(full_data[key].shape) == 3:
            eeg_key = key
            break
    
    if eeg_key is None:
        raise ValueError("Could not find EEG data in the NPZ file")
    
    original_eeg = full_data[eeg_key]
    
    # Load reduced data if it exists
    reduced_data_path = Path('data/processed/reduced_channel_data.npz')
    if reduced_data_path.exists():
        reduced_data = np.load(reduced_data_path)
        reduced_eeg = reduced_data['eeg_data']
        top_channel_indices = reduced_data['channel_indices']
        labels = reduced_data['labels']
    else:
        reduced_eeg = None
        top_channel_indices = None
        # Try to find labels in the original data
        label_key = None
        for key in full_data.files:
            if len(full_data[key].shape) == 1 or (len(full_data[key].shape) == 2 and full_data[key].shape[1] == 1):
                label_key = key
                break
        
        if label_key:
            labels = full_data[label_key]
        else:
            labels = None
            
    return original_eeg, reduced_eeg, top_channel_indices, labels

def visualize_data_structure(original_eeg, reduced_eeg=None, top_channels=None):
    """Visualize the structure of the EEG data"""
    plt.figure(figsize=(15, 10))
    
    # Plot a heatmap of a few samples from the original data
    plt.subplot(2, 1, 1)
    sample_idx = 0  # First sample
    plt.imshow(original_eeg[sample_idx], aspect='auto', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title(f'Original EEG Data: All {original_eeg.shape[1]} Channels')
    plt.xlabel('Time Points')
    plt.ylabel('Channels')
    
    # If reduced data is available, show the same sample with only selected channels
    if reduced_eeg is not None and top_channels is not None:
        plt.subplot(2, 1, 2)
        plt.imshow(reduced_eeg[sample_idx], aspect='auto', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.title(f'Reduced EEG Data: Top {len(top_channels)} Channels {top_channels}')
        plt.xlabel('Time Points')
        plt.ylabel('Selected Channels')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'data_structure.png')
    plt.close()
    print(f"Saved data structure visualization to {VIZ_DIR / 'data_structure.png'}")

def visualize_channel_comparison(original_eeg, reduced_eeg=None, top_channels=None):
    """Visualize a comparison of signals from original vs selected channels"""
    if reduced_eeg is None or top_channels is None:
        print("Reduced data not available, skipping channel comparison")
        return
    
    sample_idx = 0  # First sample
    time_points = min(original_eeg.shape[2], 500)  # Limit to 500 time points
    
    plt.figure(figsize=(15, 12))
    
    # Plot original channels
    for i, ch_idx in enumerate(top_channels):
        plt.subplot(3, 1, i+1)
        plt.plot(original_eeg[sample_idx, ch_idx, :time_points])
        plt.title(f'Original Channel {ch_idx}')
        plt.ylabel('Amplitude')
        if i == 2:  # Only add x-label to bottom plot
            plt.xlabel('Time Points')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'channel_comparison.png')
    plt.close()
    print(f"Saved channel comparison to {VIZ_DIR / 'channel_comparison.png'}")

def recreate_channel_attention(input_shape):
    """Recreate the channel attention model for visualization"""
    inputs = layers.Input(shape=input_shape)
    
    # Channel attention mechanism
    channel_avg = layers.GlobalAveragePooling1D()(inputs)
    channel_weights = layers.Dense(input_shape[0], activation='sigmoid')(channel_avg)
    channel_weights = layers.Reshape((input_shape[0], 1))(channel_weights)
    
    # Apply channel weights
    weighted_channels = layers.Multiply()([inputs, channel_weights])
    
    # Create models
    full_model = Model(inputs=inputs, outputs=weighted_channels)
    weight_model = Model(inputs=inputs, outputs=channel_weights)
    
    return full_model, weight_model

def visualize_channel_attention_mechanism(original_eeg):
    """Visualize how the channel attention mechanism works"""
    # Get a few samples
    sample_idx = 0
    sample_data = original_eeg[sample_idx:sample_idx+3]
    
    # Recreate models
    input_shape = (original_eeg.shape[1], original_eeg.shape[2])
    full_model, weight_model = recreate_channel_attention(input_shape)
    
    # Generate random weights for visualization
    # This is just for demonstration since we don't have the trained weights
    np.random.seed(42)
    sample_weights = np.random.rand(original_eeg.shape[1])
    sample_weights = sample_weights / np.sum(sample_weights)  # Normalize
    
    plt.figure(figsize=(15, 12))
    
    # Plot channel weights - simulating what the model would learn
    ax1 = plt.subplot(3, 1, 1)
    plt.bar(range(len(sample_weights)), sample_weights)
    plt.title('Channel Attention Weights (Simulated for Visualization)')
    plt.xlabel('Channel Index')
    plt.ylabel('Weight')
    
    # Mark top 3 channels
    top3_indices = np.argsort(sample_weights)[-3:]
    for idx in top3_indices:
        plt.annotate(f'Channel {idx}', 
                     xy=(idx, sample_weights[idx]),
                     xytext=(idx, sample_weights[idx] + 0.02),
                     ha='center')
    
    # Plot original vs weighted signal for a top channel
    top_channel = top3_indices[-1]
    
    # Original signal
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(original_eeg[sample_idx, top_channel, :500])
    plt.title(f'Original Signal - Channel {top_channel}')
    plt.ylabel('Amplitude')
    
    # Weighted signal (simulated)
    ax3 = plt.subplot(3, 1, 3)
    weighted_signal = original_eeg[sample_idx, top_channel, :500] * sample_weights[top_channel]
    plt.plot(weighted_signal)
    plt.title(f'Weighted Signal - Channel {top_channel} (Weight: {sample_weights[top_channel]:.4f})')
    plt.ylabel('Weighted Amplitude')
    plt.xlabel('Time Points')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'channel_attention_mechanism.png')
    plt.close()
    print(f"Saved channel attention mechanism visualization to {VIZ_DIR / 'channel_attention_mechanism.png'}")

def visualize_channel_importance(original_eeg, top_channels=None):
    """Visualize the channel importance"""
    # Load channel importance data if the file exists
    importance_file = Path('results/channel_importance.png')
    if importance_file.exists():
        img = plt.imread(importance_file)
        
        plt.figure(figsize=(16, 10))
        
        # Plot the existing channel importance figure
        plt.subplot(2, 1, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Channel Importance from NAS')
        
        # Create a more interpretable visualization
        if top_channels is not None:
            plt.subplot(2, 1, 2)
            
            # Create a brain-like visualization of channel importance
            # This is simplified - for a real EEG visualization, you'd use a topomap
            n_channels = original_eeg.shape[1]
            importance = np.zeros(n_channels)
            for ch in top_channels:
                importance[ch] = 1
                
            # Plot the importance as a "brain" heatmap (simplified)
            plt.bar(range(n_channels), importance, color='lightblue')
            
            # Highlight the top channels
            for ch in top_channels:
                plt.bar(ch, importance[ch], color='red', label=f'Top Channel {ch}' if ch == top_channels[0] else None)
                plt.annotate(f'Ch {ch}', xy=(ch, importance[ch]+0.05), ha='center')
            
            plt.title('Top Selected Channels')
            plt.xlabel('Channel Index')
            plt.ylabel('Selected (1=Yes, 0=No)')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'channel_importance_enhanced.png')
        plt.close()
        print(f"Saved enhanced channel importance visualization to {VIZ_DIR / 'channel_importance_enhanced.png'}")
    else:
        print("Channel importance file not found, skipping visualization")

def visualize_nas_process():
    """Create a step-by-step visualization of the NAS process"""
    # Create a flowchart-like visualization
    plt.figure(figsize=(15, 12))
    
    steps = [
        "1. Load Full 64-Channel EEG Data",
        "2. Create Channel Selection Model with\nAttention Mechanism",
        "3. Train Model to Identify Important\nChannels while Learning Authentication",
        "4. Extract Channel Importance Weights",
        "5. Select Top 3 Most Important\nChannels",
        "6. Create Reduced 3-Channel Dataset",
        "7. Train Final Simplified\nAuthentication Model"
    ]
    
    for i, step in enumerate(steps):
        plt.text(0.5, 1 - (i+1)/len(steps)*0.8, step, 
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.5),
                 fontsize=14)
        
        # Add arrows between steps
        if i < len(steps) - 1:
            plt.arrow(0.5, 1 - (i+1)/len(steps)*0.8 - 0.05, 0, -0.05, 
                     head_width=0.03, head_length=0.02, fc='black', ec='black')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.savefig(VIZ_DIR / 'nas_process.png')
    plt.close()
    print(f"Saved NAS process visualization to {VIZ_DIR / 'nas_process.png'}")

def visualize_eeg_montage(top_channels=None):
    """Visualize the EEG channel montage with selected channels highlighted"""
    # This is a simplified visualization - in a real application, you'd use a proper EEG montage
    
    # Define approximate positions for 64 EEG channels in a 2D space
    # This is a simplified layout and doesn't represent any specific EEG system
    # For a real application, you'd use the standard 10-20 system coordinates
    n_channels = 64
    np.random.seed(42)  # For reproducible "random" positions
    
    # Generate positions in a circular pattern
    theta = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
    r = 0.8 + 0.2 * np.random.rand(n_channels)  # Radius with some jitter
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    plt.figure(figsize=(12, 12))
    
    # Plot all channels
    plt.scatter(x, y, s=100, c='blue', alpha=0.5, label='Regular Channels')
    
    # Highlight top channels if available
    if top_channels is not None:
        for ch in top_channels:
            plt.scatter(x[ch], y[ch], s=300, c='red', alpha=0.8)
            plt.annotate(f'Ch {ch}', xy=(x[ch], y[ch]), xytext=(x[ch]+0.05, y[ch]+0.05),
                        fontsize=14, weight='bold')
    
    # Add a "head" circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    plt.gca().add_patch(circle)
    
    # Add nose, ears for orientation
    plt.plot([0, 0], [1, 1.1], 'k-', linewidth=2)  # Nose
    plt.plot([-1.1, -1], [0, 0], 'k-', linewidth=2)  # Left ear
    plt.plot([1, 1.1], [0, 0], 'k-', linewidth=2)  # Right ear
    
    plt.title('EEG Channel Montage with Selected Channels Highlighted', fontsize=16)
    if top_channels is not None:
        plt.legend([f'Regular Channels', f'Selected Channels {top_channels}'])
    plt.axis('equal')
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.axis('off')
    
    plt.savefig(VIZ_DIR / 'eeg_montage.png')
    plt.close()
    print(f"Saved EEG montage visualization to {VIZ_DIR / 'eeg_montage.png'}")

def main():
    print("\nEEG Channel Selection Visualization")
    print("==================================")
    
    # Load data
    original_eeg, reduced_eeg, top_channels, labels = load_data()
    
    print(f"\nOriginal data shape: {original_eeg.shape}")
    if reduced_eeg is not None:
        print(f"Reduced data shape: {reduced_eeg.shape}")
    if top_channels is not None:
        print(f"Top channels selected: {top_channels}")
    
    # Run visualizations
    print("\nGenerating visualizations...")
    
    visualize_data_structure(original_eeg, reduced_eeg, top_channels)
    visualize_channel_comparison(original_eeg, reduced_eeg, top_channels)
    visualize_channel_attention_mechanism(original_eeg)
    visualize_channel_importance(original_eeg, top_channels)
    visualize_nas_process()
    visualize_eeg_montage(top_channels)
    
    print("\nAll visualizations created successfully!")
    print(f"Find all visualizations in the {VIZ_DIR} directory")

if __name__ == "__main__":
    main()