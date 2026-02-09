import numpy as np
import matplotlib.pyplot as plt
import os

# Load the dataset
print("Loading NPZ data...")
data_path = 'data/preprocessed_data.npz'
data_npz = np.load(data_path)

# Print information about each array in the file
print(f"\nNPZ file at {data_path} contains these arrays:")
for key in data_npz.files:
    arr = data_npz[key]
    print(f"- '{key}': shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")

# Try to identify the EEG data array
eeg_candidates = []
for key in data_npz.files:
    arr = data_npz[key]
    if len(arr.shape) == 3:
        eeg_candidates.append((key, arr.shape))

print(f"\nPotential EEG data arrays (3D arrays):")
for i, (key, shape) in enumerate(eeg_candidates):
    print(f"{i+1}. '{key}' with shape {shape}")

# If we found at least one candidate, analyze it
if eeg_candidates:
    # Select the first candidate for analysis (modify if needed)
    key_to_analyze = eeg_candidates[0][0]
    eeg_data = data_npz[key_to_analyze]
    
    # Determine which dimensions are likely samples, channels, and timepoints
    shape = eeg_data.shape
    dims = sorted(shape)  # Sort dimensions by size
    
    print(f"\nAnalyzing array '{key_to_analyze}' with shape {shape}")
    
    # Expected structure for EEG data
    if shape[0] < shape[1] and shape[0] < shape[2]:
        print("WARNING: First dimension is smallest, which is unusual for EEG data.")
        print("Expected structure: [samples, channels, timepoints]")
    
    # Check if any dimension is suspiciously large for EEG channels
    if shape[1] > 256:  # Most EEG systems have < 256 channels
        print(f"WARNING: Second dimension has {shape[1]} elements, which is unusually large for EEG channels.")
        print("Typical EEG datasets have 1-256 channels.")
    
    # Visualize data distribution
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribution of values across entire dataset
    plt.subplot(2, 2, 1)
    plt.hist(eeg_data.flatten(), bins=50)
    plt.title(f"Distribution of all values in '{key_to_analyze}'")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    # Plot 2: Example of one sample across all channels
    plt.subplot(2, 2, 2)
    sample_idx = 0
    plt.imshow(eeg_data[sample_idx], aspect='auto', cmap='viridis')
    plt.title(f"Sample {sample_idx}: All channels × timepoints")
    plt.xlabel("Timepoints")
    plt.ylabel("Channels")
    plt.colorbar(label="Amplitude")
    
    # Plot 3: Signal from a few channels for one sample
    plt.subplot(2, 2, 3)
    # Pick channels based on dataset size
    if shape[1] <= 10:
        channels_to_plot = range(shape[1])
    else:
        channels_to_plot = np.linspace(0, shape[1]-1, 5, dtype=int)
    
    for ch in channels_to_plot:
        plt.plot(eeg_data[sample_idx, ch], label=f"Channel {ch}")
    plt.title(f"Sample {sample_idx}: Signal from selected channels")
    plt.xlabel("Timepoints")
    plt.ylabel("Amplitude")
    plt.legend()
    
    # Plot 4: Variance across channels (to identify active vs. inactive channels)
    plt.subplot(2, 2, 4)
    channel_variance = np.var(eeg_data, axis=(0, 2))
    plt.semilogy(channel_variance)  # Log scale for better visibility
    plt.title("Variance across channels")
    plt.xlabel("Channel index")
    plt.ylabel("Variance (log scale)")
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/data_structure_analysis.png')
    print("Saved analysis plots to 'results/data_structure_analysis.png'")
    
    # Check for potential transpose issues
    if shape[1] > 1000:  # If "channel" dimension is very large
        print("\nPossible axis transposition detected!")
        print("Your data might be in format [samples, timepoints, channels]")
        print("or [channels, timepoints, samples] instead of [samples, channels, timepoints]")
        
        # Look at alternative interpretation of the data
        transposed_view = np.transpose(eeg_data, (0, 2, 1))
        print(f"If transposed to [samples, timepoints, channels]: shape={transposed_view.shape}")
        
        # Save a visualization of the transposed view
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(eeg_data[0], aspect='auto')
        plt.title("Original first sample")
        plt.xlabel("Timepoints" if shape[1] < shape[2] else "Channels")
        plt.ylabel("Channels" if shape[1] < shape[2] else "Timepoints")
        
        plt.subplot(1, 2, 2)
        plt.imshow(transposed_view[0], aspect='auto')
        plt.title("Transposed first sample")
        plt.xlabel("Channels")
        plt.ylabel("Timepoints")
        
        plt.tight_layout()
        plt.savefig('results/transposed_view.png')
        print("Saved transposed view analysis to 'results/transposed_view.png'")
        
    # Check for sparse or zero-filled data
    zero_percentage = np.sum(eeg_data == 0) / eeg_data.size * 100
    if zero_percentage > 50:
        print(f"\nWARNING: {zero_percentage:.1f}% of values are exactly zero, which is unusually high.")
    
    # Check if transposing would give more reasonable channel numbers
    if shape[1] > 256 and shape[2] < 256:
        print("\nSuggestion: Try transposing your data to swap channels and timepoints:")
        print("eeg_data = np.transpose(eeg_data, (0, 2, 1))")
else:
    print("\nNo 3D arrays found in the NPZ file. EEG data should be 3D [samples, channels, timepoints].")