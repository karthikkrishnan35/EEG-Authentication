import numpy as np
import scipy.signal as signal
import os

def load_raw_data(file_path):
    """Load raw EEG data from a specified file path."""
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def filter_eeg_data(eeg_data, lowcut=1.0, highcut=50.0, fs=160.0, order=5):
    """Apply a bandpass filter to the EEG data."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, eeg_data, axis=1)
    return filtered_data

def normalize_eeg_data(eeg_data):
    """Normalize the EEG data to have zero mean and unit variance."""
    return (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / np.std(eeg_data, axis=1, keepdims=True)

def segment_eeg_data(eeg_data, segment_length=160):
    """Segment the EEG data into smaller chunks."""
    segments = []
    num_segments = eeg_data.shape[1] // segment_length
    for i in range(num_segments):
        segment = eeg_data[:, i * segment_length:(i + 1) * segment_length]
        segments.append(segment)
    return np.array(segments)

def preprocess_eeg(file_path):
    """Main function to preprocess EEG data."""
    raw_data = load_raw_data(file_path)
    filtered_data = filter_eeg_data(raw_data)
    normalized_data = normalize_eeg_data(filtered_data)
    segmented_data = segment_eeg_data(normalized_data)
    return segmented_data