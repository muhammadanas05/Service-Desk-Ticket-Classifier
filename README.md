# Service Desk Ticket Classifier

## Project Overview

This project implements a Convolutional Neural Network (CNN) for classifying service desk tickets into predefined categories. The goal is to enhance the efficiency of customer service operations by automating ticket categorization.

## Description

### 1. Data Preparation:
     - TicketDataset Class: Created a custom dataset class to handle ticket data.
     - Initialization: Loads ticket texts, labels, and a word-to-index mapping from JSON and Numpy files.
     - Text Processing: Converts text data into indices and pads or truncates sequences to ensure uniform length.
     - Data Retrieval: Retrieves text and label pairs for each sample, converting them into PyTorch tensors.

     - Data Loaders: Utilized `DataLoader` to facilitate batch processing of training and testing data.
     - Training Data Loader: `train_loader` with shuffling enabled for model training.
     - Testing Data Loader: `test_loader` for evaluating model performance.

### 2. Model Definition:
     - CNNClassifier Class: Defined a CNN model for text classification.
     - Embedding Layer: Embeds word indices into dense vectors of a specified dimension.
     - 1D Convolution Layer: Applies 1D convolutions to capture features from the embedded sequences.
     - Adaptive Max Pooling: Reduces the dimensionality of the feature maps to a fixed size.
     - Linear Layer: Outputs class probabilities for the ticket categories.

### 3. Model Training:
   - Configured the model with a loss function (Cross Entropy Loss) and optimizer (Adam).
   - Training Loop: Trained the model for 3 epochs using the training data.
     - Forward Pass: Computes predictions for the input batch.
     - Backward Pass: Computes gradients and updates model parameters based on the loss.

### 4. Model Evaluation:
   - Testing Phase: Evaluated the model using the test data.
     - Predictions: Generated predictions for each test sample.
     - Metrics Calculation: Computed performance metrics including accuracy, precision, and recall.

### 5. Metrics:
   - Accuracy: Overall correctness of the model’s predictions.
   - Per-Class Precision: Precision score for each class.
   - Per-Class Recall: Recall score for each class.
   - Metrics Storage: Saved computed metrics in a file (`metrics01.pth`) for later analysis.
