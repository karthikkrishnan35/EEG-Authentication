import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns

# Make results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Load the reduced channel data
print("Loading reduced 3-channel data...")
data = np.load('data/processed/reduced_channel_data.npz')
eeg_data = data['eeg_data']
channel_indices = data['channel_indices']
labels = data['labels']

print(f"Data shape: {eeg_data.shape}")
print(f"Selected channel indices: {channel_indices}")
print(f"Labels shape: {labels.shape}")
print(f"Number of authorized samples: {np.sum(labels)}")
print(f"Number of imposter samples: {len(labels) - np.sum(labels)}")

# Define frequency bands
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

# Sampling rate (adjust if different)
sampling_rate = 128  # Hz

def extract_features(data, fs=sampling_rate):
    """Extract features from EEG data"""
    n_samples, n_channels, n_timepoints = data.shape
    features = []
    
    for sample_idx in range(n_samples):
        sample_features = []
        
        for ch_idx in range(n_channels):
            channel_data = data[sample_idx, ch_idx, :]
            
            # Time domain features
            mean = np.mean(channel_data)
            variance = np.var(channel_data)
            std_dev = np.std(channel_data)
            max_value = np.max(channel_data)
            min_value = np.min(channel_data)
            peak_to_peak = max_value - min_value
            skewness = skew(channel_data)
            kurt = kurtosis(channel_data)
            
            # Frequency domain features
            freqs, psd = welch(channel_data, fs=fs, nperseg=min(256, len(channel_data)))
            
            # Calculate band powers
            band_powers = {}
            for band_name, (low_freq, high_freq) in bands.items():
                idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                band_power = np.sum(psd[idx_band])
                band_powers[band_name] = band_power
            
            # Normalized band powers
            total_power = np.sum(psd)
            norm_band_powers = {band: power/total_power for band, power in band_powers.items()}
            
            # Band power ratios
            alpha_beta_ratio = band_powers['alpha'] / (band_powers['beta'] + 1e-10)
            theta_beta_ratio = band_powers['theta'] / (band_powers['beta'] + 1e-10)
            
            # Combine all features for this channel
            channel_features = [
                mean, variance, std_dev, peak_to_peak, skewness, kurt,
                norm_band_powers['delta'], norm_band_powers['theta'], 
                norm_band_powers['alpha'], norm_band_powers['beta'], 
                norm_band_powers['gamma'],
                alpha_beta_ratio, theta_beta_ratio
            ]
            
            sample_features.extend(channel_features)
        
        # Add cross-channel features
        if n_channels > 1:
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    # Simple correlation between channels
                    corr = np.corrcoef(data[sample_idx, i, :], data[sample_idx, j, :])[0, 1]
                    sample_features.append(corr)
        
        features.append(sample_features)
    
    return np.array(features)

print("\nExtracting features...")
features = extract_features(eeg_data)
print(f"Feature matrix shape: {features.shape}")

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Save features
os.makedirs('data/features', exist_ok=True)
np.savez('data/features/three_channel_features.npz',
         features=scaled_features,
         labels=labels)
print("Features saved to 'data/features/three_channel_features.npz'")

# Split data for classification
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, labels, test_size=0.2, stratify=labels, random_state=42
)

# Train and evaluate multiple classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', C=1, probability=True, random_state=42)
}

# Evaluate with cross-validation
print("\nCross-validation results:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"{name}: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Evaluate on test set
results = {}
plt.figure(figsize=(15, 10))

for i, (name, clf) in enumerate(classifiers.items()):
    # Train on training data
    clf.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'accuracy': acc,
        'auc': auc,
        'confusion_matrix': cm
    }
    
    # Plot confusion matrix
    plt.subplot(2, 2, i+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Imposter', 'Authorized'],
                yticklabels=['Imposter', 'Authorized'])
    plt.title(f'{name} (Acc: {acc:.4f}, AUC: {auc:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('results/classification_results_3channels.png')
print("Classification results saved to 'results/classification_results_3channels.png'")

# Print final results
print("\nTest set results:")
for name, metrics in results.items():
    print(f"{name}: Accuracy = {metrics['accuracy']:.4f}, AUC = {metrics['auc']:.4f}")

# Calculate Equal Error Rate (EER) for the best classifier
# (SVM usually works well for this task)
best_clf = classifiers['SVM (RBF)']
best_clf.fit(X_train, y_train)
y_prob = best_clf.predict_proba(X_test)[:, 1]

# Calculate FPR and TPR at various thresholds
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
fnr = 1 - tpr

# Find the threshold where FPR = FNR (Equal Error Rate)
eer_idx = np.argmin(np.abs(fpr - fnr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
eer_threshold = thresholds[eer_idx]

print(f"\nEqual Error Rate (EER): {eer:.4f} at threshold {eer_threshold:.4f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["SVM (RBF)"]["auc"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.plot(fpr[eer_idx], tpr[eer_idx], 'ro', label=f'EER = {eer:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for 3-Channel Authentication')
plt.legend()
plt.grid(True)
plt.savefig('results/roc_curve_3channels.png')
print("ROC curve saved to 'results/roc_curve_3channels.png'")

print("\nFeature extraction and classification complete!")