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

Ensure you have the following installed:
- Python 3.x
- TensorFlow 2.x
- NumPy
- MNE
- scikit-learn
- Matplotlib

## Installation

1. Clone the repository:
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
    - Split the data into training and validation sets.

2. **Define EEGNet Model**:
    - Create the EEGNet model architecture.
    - Compile the model with an appropriate optimizer, loss function, and metrics.

3. **Train EEGNet Model**:
    - Train the EEGNet model using the training data.
    - Validate the model using the validation data.

4. **Define TSCeption Model**:
    - Create the TSCeption model architecture.
    - Compile the model with an appropriate optimizer, loss function, and metrics.

5. **Train TSCeption Model**:
    - Train the TSCeption model using the training data.
    - Validate the model using the validation data.

6. **Evaluate Models**:
    - Use the validation data to evaluate the models.
    - Calculate metrics such as accuracy, precision, recall, and F1-score.

## Model Architecture

### EEGNet

EEGNet is designed specifically for EEG signal classification tasks. It uses convolutional layers to capture spatial and temporal features from the EEG signals.

### TSCeption

TSCeption is designed for time-series classification. It uses multiple convolutional layers with different kernel sizes to capture features at various temporal scales.

## Training and Validation

- Train the models on the training data.
- Validate the models on the validation data.
- Adjust hyperparameters as needed to optimize performance.

## Evaluation Metrics

Evaluate the models using the following metrics:
- **Accuracy**: Overall correctness of the model’s predictions.
- **Precision**: Proportion of true positive predictions out of all positive predictions made by the model.
- **Recall**: Proportion of true positive predictions out of all actual positive instances.
- **F1-score**: Harmonic mean of precision and recall.

## Acknowledgments

- Thanks to the PhysioNet team for providing the Mental Arithmetic Tasks Dataset.
- The EEGNet and TSCeption model architectures are based on research papers and existing implementations.

---

This README provides a comprehensive guide for anyone looking to understand and replicate your project on EEG binary classification using EEGNet and TSCeption models.
