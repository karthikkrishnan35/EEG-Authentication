import numpy as np

# Load the reduced EEG data
data = np.load("data/processed/reduced_channel_data.npz")

# Check available keys
print("Keys in NPZ file:", data.files)

# Extract EEG data and labels using correct keys
X = data['eeg_data']  # EEG data (samples, channels, timepoints)
y = data['labels']    # Labels (e.g., subject ID or authentication labels)

# Print data shapes
print(f"EEG Data Shape: {X.shape}")  # Expected: (samples, channels, timepoints)
print(f"Labels Shape: {y.shape}")    # Expected: (samples,)
