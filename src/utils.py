import numpy as np
import os
import matplotlib.pyplot as plt

def load_eeg_data(file_path):
    """Load EEG data from a .npy file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return np.load(file_path)

def save_eeg_data(data, file_path):
    """Save EEG data to a .npy file."""
    np.save(file_path, data)

def plot_eeg_signals(eeg_data, sampling_rate=160, duration=5):
    """Plot EEG signals for the first 'duration' seconds."""
    time = np.linspace(0, duration, duration * sampling_rate)
    plt.figure(figsize=(10, 6))
    for i in range(eeg_data.shape[0]):
        plt.plot(time, eeg_data[i, : len(time)] + i * 200, label=f"Channel {i+1}")
    plt.title("EEG Signals (First 5 Seconds, Scaled)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (shifted)")
    plt.legend()
    plt.show()