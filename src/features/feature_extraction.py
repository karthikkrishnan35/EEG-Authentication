import numpy as np

def extract_features(eeg_data):
    features = {}
    
    # Example feature: Mean of each channel
    features['mean'] = np.mean(eeg_data, axis=1)
    
    # Example feature: Standard deviation of each channel
    features['std'] = np.std(eeg_data, axis=1)
    
    # Example feature: Maximum value of each channel
    features['max'] = np.max(eeg_data, axis=1)
    
    # Example feature: Minimum value of each channel
    features['min'] = np.min(eeg_data, axis=1)
    
    # Add more feature extraction methods as needed
    
    return features

def extract_windowed_features(eeg_data, window_size, overlap):
    num_windows = (eeg_data.shape[1] - window_size) // (window_size - overlap) + 1
    windowed_features = []

    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        window = eeg_data[:, start:end]
        features = extract_features(window)
        windowed_features.append(features)

    return windowed_features