import matplotlib.pyplot as plt
import numpy as np

# Load reduced dataset
data = np.load('data/processed/reduced_channel_data.npz')
eeg_data = data['eeg_data']  # Shape: (samples, 3, timepoints)
channel_indices = data['channel_indices']  # Selected channels

# Plot EEG signals for a few samples
num_samples_to_plot = 5  # Number of EEG samples to plot

plt.figure(figsize=(12, 6))
for i in range(num_samples_to_plot):
    plt.subplot(num_samples_to_plot, 1, i+1)
    plt.plot(eeg_data[i, 0, :], label=f"Channel {channel_indices[0]}")
    plt.plot(eeg_data[i, 1, :], label=f"Channel {channel_indices[1]}")
    plt.plot(eeg_data[i, 2, :], label=f"Channel {channel_indices[2]}")
    plt.legend()
    plt.title(f"EEG Signal Sample {i+1}")
    plt.xlabel("Timepoints")
    plt.ylabel("EEG Amplitude")

plt.tight_layout()
plt.show()
