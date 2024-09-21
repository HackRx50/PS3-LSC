# Team LSC Forgery Detection Hackrx

## Project Overview
This project focuses on **Document Forgery Detection** in the context of healthcare insurance claims. The goal is to detect digital forgeries by classifying documents as either authentic or fabricated. The project leverages deep learning models such as **CNN** and **ResNet50** architectures to perform the classification. The primary use case is to enhance fraud prevention in healthcare insurance by identifying manipulated or fake documents.

### Problem Statement
Document forgery poses a significant threat to the healthcare insurance sector, leading to financial losses and legal issues. The project aims to develop a system that can detect forgeries in submitted insurance documents by analyzing their contents and structure. The system will identify anomalies that indicate document manipulation, ensuring accuracy in the claims process.

### Project Objectives:
1. **Digital Forgery Detection**: Build a model that classifies documents as either authentic or fabricated using deep learning techniques.
2. **Accuracy**: Ensure high accuracy, precision, and recall for the forgery detection model.
3. **Model Implementation**: Focus on advanced CNN architectures such as ResNet50 to enhance detection performance.
4. **Evaluation**: Evaluate the model based on classification metrics such as accuracy, precision, recall, and F1-score.

## Team Members:
- **Saahil Shaikh** (Team Lead) 
- **Sachit Desai** 
- **Tejas Ulabhaje** 
- **Bhaavesh Waykole** 

## Approach:
1. **Data Collection & Preprocessing**: 
   - Generate and augment a dataset of sample images representing both authentic and fabricated documents.
   - Perform data preprocessing, including resizing, normalization, and augmentation, to prepare the dataset for training.

2. **Model Architecture**:
   - **Convolutional Neural Networks (CNN)**: Implement CNN models to extract features from document images.
   - **ResNet50**: Use the ResNet50 architecture to enhance feature extraction and improve classification accuracy.
   
3. **Model Training & Testing**:
   - Train the model on labeled datasets of authentic and fabricated documents.
   - Use a combination of train-test splits and cross-validation to evaluate model performance.
   - Optimize hyperparameters to improve accuracy and avoid overfitting.

4. **Evaluation**:
   - Evaluate the model based on **accuracy**, **precision**, **recall**, and **F1-score**.
   - Perform additional evaluations using confusion matrices to analyze the performance of the classification.

## Tools and Technologies:
- **Python** (TensorFlow, Keras, PyTorch)
- **Convolutional Neural Networks (CNN)**
- **ResNet50 Architecture**
- **Jupyter Notebooks** for experimentation
- **Flask** for potential API deployment
- **Docker** for containerization



