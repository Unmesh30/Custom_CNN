# Custom_CNN

## Overview

Custom_CNN is an end-to-end skin image classification project designed to determine whether an uploaded skin image is more consistent with acne-affected skin or clear skin. The project combines a deep learning inference backend built with Flask and TensorFlow and a web application built with Spring Boot and Thymeleaf.

The system was developed as a practical full-stack machine learning application in which model training, backend inference serving, and web integration are connected into a single working pipeline.

## Project Objectives

The primary objectives of this project are to:

* Build a convolutional neural network based acne classification system
* Serve trained model predictions through a lightweight Flask inference API
* Integrate the machine learning backend with a Spring Boot web application
* Allow users to upload skin images and receive prediction results through a web interface
* Evaluate and improve model behavior through dataset balancing, threshold tuning, and real-image testing

## System Architecture

The project consists of two main application components.

### Flask ML Inference Service

The Flask service is responsible for:

* Loading the trained `.keras` model
* Preprocessing uploaded skin images
* Generating prediction scores
* Returning prediction labels and threshold information through an API response

### Spring Boot Web Application

The Spring Boot application is responsible for:

* Rendering the user interface
* Handling image uploads
* Communicating with the Flask backend
* Displaying the prediction label, score, and threshold to the user

## Technology Stack

### Machine Learning

* Python
* TensorFlow / Keras
* NumPy
* PIL
* Matplotlib
* Google Colab

### Backend and Integration

* Flask
* Spring Boot
* Java
* Thymeleaf
* Maven

### Development Tools

* Visual Studio Code
* Google Drive / Colab for dataset and training workflow
* Git and GitHub for version control

## Repository Structure

```text
CUSTOM_CNN/
├── ML_SERVICES/          # Flask-based ML inference backend
├── skinpredictor/        # Spring Boot web application
├── TESTING/              # Optional local testing assets
└── README.md
```

## Machine Learning Workflow

The machine learning workflow followed in this project includes:

1. Collecting and organizing acne and clear skin image datasets
2. Balancing the dataset across classes
3. Splitting the data into training, validation, and test partitions
4. Training a custom CNN model using TensorFlow/Keras
5. Evaluating classification performance using:

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion matrix
   * ROC-AUC
6. Tuning the classification threshold to improve practical prediction behavior
7. Exporting the trained model for inference usage

## Model Performance

The refined model achieved strong evaluation performance on the held-out test set, including approximately:

* Accuracy: 0.92
* Macro F1-score: 0.92
* ROC-AUC: 0.98

Sample classification report:

```text
              precision    recall  f1-score   support

        Acne       0.91      0.93      0.92       422
       Clear       0.93      0.91      0.92       435

    accuracy                           0.92       857
   macro avg       0.92      0.92      0.92       857
weighted avg       0.92      0.92      0.92       857
```

These results indicate that the model is reasonably balanced across both classes and performs reliably on the curated dataset.

## Inference Logic

The deployed model returns a prediction score between 0 and 1.

Current practical interpretation:

* Higher score indicates stronger confidence for **Clear**
* Lower score indicates stronger confidence for **Acne**

A tuned threshold is used in deployment instead of relying only on the default 0.5 decision boundary.

## Features

* Image upload through a web interface
* Flask-based machine learning inference API
* Spring Boot integration for user interaction
* Styled dark-themed user interface
* Prediction result display with:

  * Label
  * Score
  * Decision threshold
* Modular separation between ML inference and web application layers

## How to Run the Project

### Prerequisites

Make sure the following are installed:

* Python 3.x
* Java 17 or above
* Maven
* Git

### Running the Flask ML Service

Navigate to the Flask service folder:

```bash
cd ML_SERVICES
```

Create and activate a virtual environment if needed, then install dependencies:

```bash
pip install flask tensorflow pillow numpy
```

Run the Flask application:

```bash
python app.py
```

The Flask inference service should start on:

```text
http://127.0.0.1:5000
```

### Running the Spring Boot Application

Navigate to the Spring Boot application folder:

```bash
cd skinpredictor
```

Run the application using Maven:

```bash
./mvnw spring-boot:run
```

or on Windows:

```bash
mvnw.cmd spring-boot:run
```

The Spring Boot web application should start on:

```text
http://localhost:8080
```

## Application Flow

1. The user uploads a skin image through the Spring Boot interface
2. Spring Boot sends the uploaded image to the Flask prediction service
3. Flask preprocesses the image and runs the trained CNN model
4. Flask returns the prediction response
5. Spring Boot displays the result to the user

## Current Limitations

Although the system performs well on the evaluation dataset, the following limitations remain:

* Model performance depends on image quality, framing, and lighting
* Real-world user uploads may vary significantly in angle, background, and skin visibility
* Some dataset samples may still contain noise or stylistic inconsistencies
* The system currently supports only binary classification

## Future Improvements

Planned future enhancements include:

* Improve dataset quality through hard-example curation
* Add image preview support in the frontend
* Support confidence explanations or Grad-CAM visualizations
* Improve handling of difficult real-world borderline cases
* Expand from binary classification to acne severity estimation
* Package both services for cloud deployment
* Introduce persistent logging and request tracing

## Academic and Engineering Value

This project demonstrates practical capabilities in:

* Deep learning model development
* Dataset preparation and evaluation
* Threshold-based decision tuning
* Flask API development
* Spring Boot web integration
* Full-stack AI application design

It serves as a compact but complete example of how a machine learning model can be integrated into a production-style software workflow.

## Author

Unmesh Achar

## License

This project is currently intended for academic, learning, and portfolio purposes.
