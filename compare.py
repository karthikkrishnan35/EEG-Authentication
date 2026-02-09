# Load original EEG data
import numpy as np
import matplotlib.pyplot as plt
import os

original_data = np.load('data/preprocessed_data_transposed.npz')

# Select a sample for visualization
sample_idx = 0

plt.figure(figsize=(12, 6))

# Original EEG Plot (64 Channels)
plt.subplot(2, 1, 1)
plt.imshow(original_data[sample_idx, :, :], aspect='auto', cmap='jet')
plt.colorbar()
plt.title("Original EEG Data (All Channels)")
plt.ylabel("Channels")
plt.xlabel("Timepoints")

# Reduced EEG Plot (3 Channels)
plt.subplot(2, 1, 2)
plt.imshow(eeg_data[sample_idx, :, :], aspect='auto', cmap='jet')
plt.colorbar()
plt.title("Reduced EEG Data (Top 3 Channels)")
plt.ylabel("Channels")
plt.xlabel("Timepoints")

plt.tight_layout()
plt.show()
