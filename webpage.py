from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import os
import json
import base64
import io
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the model
model = None

@app.before_first_request
def load_model():
    global model
    try:
        model = tf.keras.models.load_model('../eeg_auth_model.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

# Feature extraction function (same as in your training code)
def extract_features(eeg_data, sampling_rate=128):
    # Implement the same feature extraction as in your training code
    # This is a placeholder - replace with your actual feature extraction
    
    # Example: simple features
    features = []
    for channel in range(eeg_data.shape[0]):
        channel_data = eeg_data[channel]
        
        # Time domain features
        mean = np.mean(channel_data)
        variance = np.var(channel_data)
        
        # Frequency domain features (using FFT)
        freqs, psd = signal.welch(channel_data, fs=sampling_rate, nperseg=256)
        
        # Extract band powers (example)
        delta_idx = np.logical_and(freqs >= 1, freqs <= 4)
        theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
        alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
        beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
        
        delta_power = np.sum(psd[delta_idx])
        theta_power = np.sum(psd[theta_idx])
        alpha_power = np.sum(psd[alpha_idx])
        beta_power = np.sum(psd[beta_idx])
        
        # Combine features
        features.extend([mean, variance, delta_power, theta_power, alpha_power, beta_power])
    
    return np.array(features)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get data from request
        data = request.get_json()
        eeg_data = np.array(data['eeg_data'])
        
        # Extract features
        features = extract_features(eeg_data)
        features = features.reshape(1, -1)  # Reshape for model input
        
        # Get prediction
        prediction = model.predict(features)[0][0]
        is_authenticated = bool(prediction > 0.5)
        
        # Generate EEG plot
        fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for i in range(3):
            ax[i].plot(eeg_data[i])
            ax[i].set_ylabel(f"Channel {i+1}")
        ax[2].set_xlabel("Samples")
        
        # Convert plot to base64 for sending to client
        buf = io.BytesIO()
        fig.tight_layout()
        FigureCanvasAgg(fig).print_png(buf)
        plt.close(fig)
        
        plot_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        
        return jsonify({
            'authenticated': is_authenticated,
            'confidence': float(prediction),
            'plot': plot_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)