import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocess import preprocess_data
from src.features.feature_extraction import extract_features
from src.models.model import create_model

def train_model(data_path, model_save_path, test_size=0.2, random_state=42):
    # Load and preprocess the data
    raw_data = np.load(data_path)
    preprocessed_data = preprocess_data(raw_data)

    # Extract features
    features, labels = extract_features(preprocessed_data)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    # Create the model
    model = create_model()

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_save_path)

if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "processed_eeg_data.npy")
    model_save_path = os.path.join("models", "eeg_auth_model.pkl")
    train_model(data_path, model_save_path)