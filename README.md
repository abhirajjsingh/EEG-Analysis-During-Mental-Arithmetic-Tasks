# EEG-Analysis-During-Mental-Arithmetic-Tasks

# EEG Binary Classification using EEGNet and TSCeption

This project involves binary classification of EEG data using deep learning models, specifically EEGNet and TSCeption. The dataset used is the Mental Arithmetic Tasks Dataset from PhysioNet. The project includes data preprocessing, feature extraction, model training, validation, and evaluation using metrics like accuracy, precision, recall, and F1-score.

## Table of Contents

- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
  - [EEGNet](#eegnet)
  - [TSCeption](#tsception)
- [Training and Validation](#training-and-validation)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Dataset

The dataset used in this project is the Mental Arithmetic Tasks Dataset from PhysioNet.

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy
- MNE
- scikit-learn
- Matplotlib

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/eeg-binary-classification.git
    cd eeg-binary-classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Load and Preprocess Data**:
    - Load the EEG data for rest and task states.
    - Preprocess the data (e.g., filtering, epoching, normalization).

    ```python
    import mne
    import numpy as np

    # Load EEG data
    eeg_rest_path = 'path_to_rest_state_data.edf'
    eeg_task_path = 'path_to_task_state_data.edf'

    raw_rest = mne.io.read_raw_edf(eeg_rest_path, preload=True)
    raw_task = mne.io.read_raw_edf(eeg_task_path, preload=True)

    # Preprocess Data (e.g., filtering, epoching)
    # Apply band-pass filter
    raw_rest.filter(1., 100., fir_design='firwin')
    raw_task.filter(1., 100., fir_design='firwin')

    # Extract epochs
    events_rest = mne.make_fixed_length_events(raw_rest, duration=2)
    events_task = mne.make_fixed_length_events(raw_task, duration=2)

    epochs_rest = mne.Epochs(raw_rest, events_rest, tmin=0, tmax=2, baseline=None, preload=True)
    epochs_task = mne.Epochs(raw_task, events_task, tmin=0, tmax=2, baseline=None, preload=True)

    X_rest = epochs_rest.get_data()
    X_task = epochs_task.get_data()

    y_rest = np.zeros(len(X_rest))
    y_task = np.ones(len(X_task))

    X = np.concatenate((X_rest, X_task), axis=0)
    y = np.concatenate((y_rest, y_task), axis=0)

    # Split data into train and validation sets
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

2. **Define EEGNet Model**:

    ```python
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout

    def EEGNet_Model(input_shape):
        model = Sequential()
        model.add(Conv2D(16, (1, 64), input_shape=input_shape, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (2, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(1, 4)))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

    input_shape = (X_train.shape[1], X_train.shape[2], 1)  # Adjust according to your data

    eegnet_model = EEGNet_Model(input_shape)

    eegnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ```

3. **Train EEGNet Model**:

    ```python
    history_eegnet = eegnet_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    ```

4. **Define TSCeption Model**:

    ```python
    from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D

    def TSCeption_Model(input_shape):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))
        return model

    input_shape = (X_train.shape[2], X_train.shape[1])  # Adjust according to your data

    tsception_model = TSCeption_Model(input_shape)

    tsception_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ```

5. **Train TSCeption Model**:

    ```python
    history_tsception = tsception_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    ```

6. **Evaluate Models**:

    ```python
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred_eegnet = (eegnet_model.predict(X_val) > 0.5).astype('int32')
    y_pred_tsception = (tsception_model.predict(X_val) > 0.5).astype('int32')

    # EEGNet Metrics
    accuracy_eegnet = accuracy_score(y_val, y_pred_eegnet)
    precision_eegnet = precision_score(y_val, y_pred_eegnet)
    recall_eegnet = recall_score(y_val, y_pred_eegnet)
    f1_eegnet = f1_score(y_val, y_pred_eegnet)

    # TSCeption Metrics
    accuracy_tsception = accuracy_score(y_val, y_pred_tsception)
    precision_tsception = precision_score(y_val, y_pred_tsception)
    recall_tsception = recall_score(y_val, y_pred_tsception)
    f1_tsception = f1_score(y_val, y_pred_tsception)

    print("EEGNet Model Metrics:")
    print(f"Accuracy: {accuracy_eegnet:.4f}")
    print(f"Precision: {precision_eegnet:.4f}")
    print(f"Recall: {recall_eegnet:.4f}")
    print(f"F1-score: {f1_eegnet:.4f}")

    print("\nTSCeption Model Metrics:")
    print(f"Accuracy: {accuracy_tsception:.4f}")
    print(f"Precision: {precision_tsception:.4f}")
    print(f"Recall: {recall_tsception:.4f}")
    print(f"F1-score: {f1_tsception:.4f}")
    ```

## Model Architecture

### EEGNet

EEGNet is designed specifically for EEG signal classification tasks. It uses convolutional layers to capture spatial and temporal features from the EEG signals.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense

def EEGNet_Model(input_shape):
    model = Sequential()
    model.add(Conv2D(16, (1, 64), input_shape=input_shape, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (2, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 4)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


-- Evaluation Metrics
Evaluate the models using the following metrics:

Accuracy: Overall correctness of the model’s predictions.
Precision: Proportion of true positive predictions out of all positive predictions made by the model.
Recall: Proportion of true positive predictions out of all actual positive instances.
F1-score: Harmonic mean of precision and recall.
