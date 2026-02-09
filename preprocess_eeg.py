import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy import signal

# Paths and parameters
data_dir = 'data/raw'
output_dir = 'data/preprocessed'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

# Processing parameters
sample_rate = 128  # Target sample rate (Hz)
bandpass_filter = (1, 40)  # Hz - common EEG range
notch_filter = 50  # Hz - power line noise (use 60 if in USA)

def preprocess_and_visualize(file_path):
    """Preprocess EDF file with visualization of each step"""
    print(f"Processing file: {file_path}")
    
    # Create a figure for visualization
    fig = plt.figure(figsize=(15, 12))
    
    # Step 1: Load EDF file
    raw = mne.io.read_raw_edf(file_path, preload=True)
    print(f"Original sample rate: {raw.info['sfreq']} Hz")
    print(f"Channels: {raw.ch_names}")
    print(f"Data shape: {raw.get_data().shape}")
    
    # Visualize original data
    ax1 = fig.add_subplot(5, 1, 1)
    data_orig = raw.get_data()
    # Plot first 5 channels for the first 5 seconds
    for i in range(min(5, data_orig.shape[0])):
        ax1.plot(data_orig[i, :int(5*raw.info['sfreq'])] + i*100, label=f'Ch {i}')
    ax1.set_title('Original EEG Data (First 5 seconds)', fontsize=12)
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right')
    
    # Step 2: Resample if needed
    if raw.info['sfreq'] != sample_rate:
        raw.resample(sample_rate)
        print(f"Resampled to {sample_rate} Hz")
    
    # Visualize resampled data
    ax2 = fig.add_subplot(5, 1, 2)
    data_resampled = raw.get_data()
    # Plot first 5 channels for the first 5 seconds
    for i in range(min(5, data_resampled.shape[0])):
        ax2.plot(data_resampled[i, :int(5*sample_rate)] + i*100, label=f'Ch {i}')
    ax2.set_title(f'Resampled EEG Data ({sample_rate} Hz)', fontsize=12)
    ax2.set_ylabel('Amplitude')
    
    # Step 3: Apply band-pass filter
    raw_filtered = raw.copy()
    raw_filtered.filter(bandpass_filter[0], bandpass_filter[1])
    print(f"Applied bandpass filter ({bandpass_filter[0]}-{bandpass_filter[1]} Hz)")
    
    # Visualize band-pass filtered data
    ax3 = fig.add_subplot(5, 1, 3)
    data_bandpassed = raw_filtered.get_data()
    # Plot first 5 channels for the first 5 seconds
    for i in range(min(5, data_bandpassed.shape[0])):
        ax3.plot(data_bandpassed[i, :int(5*sample_rate)] + i*100, label=f'Ch {i}')
    ax3.set_title(f'Band-pass Filtered EEG ({bandpass_filter[0]}-{bandpass_filter[1]} Hz)', fontsize=12)
    ax3.set_ylabel('Amplitude')
    
    # Step 4: Apply notch filter
    raw_filtered.notch_filter(notch_filter)
    print(f"Applied notch filter at {notch_filter} Hz")
    
    # Visualize notch filtered data
    ax4 = fig.add_subplot(5, 1, 4)
    data_notched = raw_filtered.get_data()
    # Plot first 5 channels for the first 5 seconds
    for i in range(min(5, data_notched.shape[0])):
        ax4.plot(data_notched[i, :int(5*sample_rate)] + i*100, label=f'Ch {i}')
    ax4.set_title(f'Notch Filtered EEG (Removed {notch_filter} Hz)', fontsize=12)
    ax4.set_ylabel('Amplitude')
    
    # Step 5: Perform ICA for artifact removal (blinks, muscle, etc.)
    try:
        # First, create epochs to better detect artifacts
        events = mne.make_fixed_length_events(raw_filtered, duration=1.0)
        epochs = mne.Epochs(raw_filtered, events, tmin=0, tmax=1.0, baseline=None, preload=True)
        
        # ICA
        n_components = min(15, len(raw_filtered.ch_names) - 1)  # Ensure we don't use too many components
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
        ica.fit(epochs)
        
        # Find components that correlate with EOG (eye movements)
        # In a real pipeline, you'd identify these components from EOG channels
        # For this example, we'll just exclude components that look like eye blinks (typically first component)
        if len(ica.exclude) == 0 and n_components > 0:  # Only exclude if not already excluded
            ica.exclude = [0]  # Exclude the first component as an example
        
        # Apply ICA to remove artifacts
        epochs_cleaned = epochs.copy()
        ica.apply(epochs_cleaned)
        
        # Get average of cleaned epochs
        data_cleaned = epochs_cleaned.get_data().mean(axis=0)  # Average across epochs
    except Exception as e:
        print(f"ICA cleaning failed: {e}. Using filtered data instead.")
        # If ICA fails, just use the filtered data
        data_cleaned = data_notched
    
    # Visualize cleaned data
    ax5 = fig.add_subplot(5, 1, 5)
    # Plot first 5 channels
    for i in range(min(5, data_notched.shape[0])):
        ax5.plot(data_notched[i, :int(5*sample_rate)] + i*100, label=f'Ch {i}')
    ax5.set_title('Cleaned EEG Data', fontsize=12)
    ax5.set_xlabel('Time (samples)')
    ax5.set_ylabel('Amplitude')
    
    plt.tight_layout()
    
    # Save figure showing preprocessing steps
    file_name = os.path.basename(file_path).split('.')[0]
    fig_path = os.path.join(output_dir, 'figures', f"{file_name}_preprocessing.png")
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Preprocessing visualization saved to {fig_path}")
    
    # Also create and save frequency domain visualization
    fig2, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Calculate and plot PSD for original data
    f, psd_original = signal.welch(data_orig.mean(axis=0), fs=raw.info['sfreq'], nperseg=1024)
    axes[0].semilogy(f, psd_original)
    axes[0].set_title('Power Spectral Density - Original Data', fontsize=12)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power/Frequency (dB/Hz)')
    axes[0].grid(True)
    
    # Calculate and plot PSD for cleaned data
    f, psd_cleaned = signal.welch(data_notched.mean(axis=0), fs=sample_rate, nperseg=1024)
    axes[1].semilogy(f, psd_cleaned)
    axes[1].set_title('Power Spectral Density - Cleaned Data', fontsize=12)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power/Frequency (dB/Hz)')
    axes[1].grid(True)
    
    plt.tight_layout()
    fig_freq_path = os.path.join(output_dir, 'figures', f"{file_name}_frequency_analysis.png")
    plt.savefig(fig_freq_path)
    plt.close(fig2)
    print(f"Frequency analysis saved to {fig_freq_path}")
    
    # Return the final cleaned data
    return data_notched, raw_filtered.info

def process_subject(subject_id):
    """Process all EDF files for a specific subject"""
    subject_dir = os.path.join(data_dir, subject_id)
    
    if os.path.isdir(subject_dir):
        print(f"\n==== Processing subject: {subject_id} ====")
        
        # Get list of EDF files for this subject
        edf_files = [f for f in os.listdir(subject_dir) if f.endswith('.edf')]
        
        if not edf_files:
            print(f"No EDF files found for subject {subject_id}")
            return
        
        all_data = []
        channel_names = None
        
        for file in edf_files:
            file_path = os.path.join(subject_dir, file)
            cleaned_data, info = preprocess_and_visualize(file_path)
            
            # Store data and channel info
            all_data.append(cleaned_data)
            if channel_names is None:
                channel_names = info['ch_names']
        
        # Combine all recordings for this subject
        combined_data = np.concatenate(all_data, axis=1) if all_data else np.array([])
        print(f"Combined data shape: {combined_data.shape}")
        
        # Save preprocessed data
        np.savez(os.path.join(output_dir, f"{subject_id}_preprocessed.npz"),
                 eeg_data=combined_data,
                 channel_names=channel_names,
                 sampling_rate=sample_rate)
        
        print(f"Preprocessed data saved to {os.path.join(output_dir, f'{subject_id}_preprocessed.npz')}")
        
        # Generate additional visualization of the combined data
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        # Plot first 5 channels for 10 seconds
        seconds_to_plot = 10
        samples_to_plot = seconds_to_plot * sample_rate
        for i in range(min(5, combined_data.shape[0])):
            plt.plot(combined_data[i, :min(samples_to_plot, combined_data.shape[1])] + i*100, 
                     label=channel_names[i] if channel_names and i < len(channel_names) else f'Ch {i}')
        plt.title(f'Preprocessed EEG Data - First {seconds_to_plot} seconds', fontsize=14)
        plt.ylabel('Amplitude')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        # Plot channel correlation matrix
        corr_matrix = np.corrcoef(combined_data)
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title('Channel Correlation Matrix', fontsize=14)
        tick_positions = np.arange(0, combined_data.shape[0], 
                                  max(1, combined_data.shape[0]//20))  # Show up to 20 tick labels
        plt.xticks(tick_positions)
        plt.yticks(tick_positions)
        plt.xlabel('Channel Index')
        plt.ylabel('Channel Index')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', f"{subject_id}_overview.png"))
        plt.close()
        
        print(f"Overview visualization saved to {os.path.join(output_dir, 'figures', f'{subject_id}_overview.png')}")
        
        # Return success
        return True
    else:
        print(f"Subject directory not found: {subject_dir}")
        return False

def main():
    """Process all subjects in the raw data directory"""
    # Get list of all subject directories
    subject_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not subject_dirs:
        print(f"No subject directories found in {data_dir}")
        return
    
    print(f"Found {len(subject_dirs)} subjects: {subject_dirs}")
    
    # Process each subject
    processed_subjects = 0
    for subject_id in subject_dirs:
        if process_subject(subject_id):
            processed_subjects += 1
    
    print(f"\nPreprocessing complete. Processed {processed_subjects} subjects.")
    
    # Create a summary file with information about all subjects
    with open(os.path.join(output_dir, 'preprocessing_summary.txt'), 'w') as f:
        f.write(f"EEG Preprocessing Summary\n")
        f.write(f"=======================\n\n")
        f.write(f"Total subjects processed: {processed_subjects}\n")
        f.write(f"Sampling rate: {sample_rate} Hz\n")
        f.write(f"Bandpass filter: {bandpass_filter[0]}-{bandpass_filter[1]} Hz\n")
        f.write(f"Notch filter: {notch_filter} Hz\n\n")
        
        f.write(f"Subject Details:\n")
        
        # Add details for each processed subject
        for subject_id in subject_dirs:
            npz_path = os.path.join(output_dir, f"{subject_id}_preprocessed.npz")
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                eeg_data = data['eeg_data']
                f.write(f"- {subject_id}: {eeg_data.shape[0]} channels, {eeg_data.shape[1]} samples "
                        f"({eeg_data.shape[1]/sample_rate:.2f} seconds)\n")
    
    print(f"Summary saved to {os.path.join(output_dir, 'preprocessing_summary.txt')}")
    
    # Create a combined visualization comparing subjects
    if processed_subjects > 0:
        plt.figure(figsize=(15, 10))
        
        # Get data from first 5 subjects (or fewer if less available)
        subjects_to_plot = subject_dirs[:min(5, len(subject_dirs))]
        
        # Plot average power spectrum for each subject
        for i, subject_id in enumerate(subjects_to_plot):
            npz_path = os.path.join(output_dir, f"{subject_id}_preprocessed.npz")
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                eeg_data = data['eeg_data']
                
                # Calculate average power spectrum across all channels
                f, psd = signal.welch(eeg_data.mean(axis=0), fs=sample_rate, nperseg=1024)
                plt.semilogy(f, psd, label=subject_id)
        
        plt.title('Power Spectral Density Comparison Between Subjects', fontsize=14)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'figures', "subjects_comparison.png"))
        plt.close()
        
        print(f"Subject comparison visualization saved to {os.path.join(output_dir, 'figures', 'subjects_comparison.png')}")

if __name__ == "__main__":
    main()