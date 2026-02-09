# EEG Authentication Project

This project aims to develop a binary classification model for EEG-based authentication. The model will utilize raw EEG data, preprocess it, extract relevant features, and evaluate its performance.

## Project Structure

- **data/**: Contains all data-related files.
  - **raw/**: Directory for raw EEG data files.
  - **processed/**: Directory for processed EEG data files after preprocessing.
  - **features/**: Directory for storing extracted features from the EEG data.

- **src/**: Contains source code for preprocessing, feature extraction, model definition, training, and evaluation.
  - **preprocessing/**: Functions for preprocessing the raw EEG data (filtering, normalization, segmentation).
    - `preprocess.py`
  - **features/**: Functions for extracting relevant features from the preprocessed EEG data.
    - `feature_extraction.py`
  - **models/**: Defines the architecture of the binary classification model and training loop.
    - `model.py`
    - `train.py`
  - **evaluation/**: Functions for evaluating the model's performance (accuracy, precision, recall, F1 score).
    - `metrics.py`
  - `utils.py`: Utility functions for data loading and visualization.

- **notebooks/**: Jupyter notebooks for exploratory data analysis, model development, and evaluation.
  - `eda.ipynb`: Exploratory data analysis of the EEG data.
  - `model_development.ipynb`: Development and testing of the binary classification model.
  - `evaluation.ipynb`: Evaluation of the trained model's performance.

- **config/**: Configuration parameters for the project.
  - `parameters.yaml`: Contains model hyperparameters and file paths.

- **tests/**: Unit tests for ensuring the functionality of preprocessing and model functions.
  - `test_preprocessing.py`: Unit tests for preprocessing functions.
  - `test_model.py`: Unit tests for model functions.

- **requirements.txt**: Lists the Python dependencies required for the project.

## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```
4. Prepare your raw EEG data and place it in the `data/raw/` directory.
5. Run the preprocessing script to process the raw data.
6. Use the Jupyter notebooks for exploratory data analysis, model development, and evaluation.

## Usage

- Use the `src/preprocessing/preprocess.py` to preprocess the raw EEG data.
- Extract features using `src/features/feature_extraction.py`.
- Train the model using `src/models/train.py`.
- Evaluate the model's performance with `src/evaluation/metrics.py`.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.