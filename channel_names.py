import mne

# Load a sample EDF file (replace with an actual file path from your dataset)
sample_edf_path = "data\S001\S001R01.edf"  # Change to a valid file path
raw = mne.io.read_raw_edf(sample_edf_path, preload=False)

# Get the list of original channel names
original_channel_names = raw.ch_names

# Selected channel indices from NAS
top_channel_indices = [21, 24, 28]  # The indices you got

# Map indices to channel names
selected_channel_names = [original_channel_names[i] for i in top_channel_indices]
print(f"Selected EEG Channel Names: {selected_channel_names}")
