import numpy as np

# Load the dataset
data_npz = np.load('data/preprocessed_data.npz')
eeg_data = data_npz['X']  # Original shape: (70, 9760, 64)
labels = data_npz['y']

# Transpose to correct format [samples, channels, timepoints]
eeg_data_transposed = np.transpose(eeg_data, (0, 2, 1))  # New shape: (70, 64, 9760)

# Save the correctly formatted data
np.savez('data/preprocessed_data_transposed.npz', 
         X=eeg_data_transposed,
         y=labels)

print(f"Original shape: {eeg_data.shape}")
print(f"Transposed shape: {eeg_data_transposed.shape}")
print("Saved transposed data to 'data/preprocessed_data_transposed.npz'")