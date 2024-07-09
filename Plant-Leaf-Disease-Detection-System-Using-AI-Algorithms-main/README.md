# Plant Leaf Disease Detection System Using AI Algorithms

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [Model Training](#model-training)
8. [Proposed Model](#proposed-model)
9. [Experimental Results](#experimental-results)
10. [Contributing](#contributing)
11. [Acknowledgements](#acknowledgements)
12. [Contact](#contact)

## Overview

This project aims to develop a system for detecting diseases in plant leaves using advanced AI algorithms. By leveraging machine learning techniques, this system can accurately identify various plant diseases, helping farmers and agriculturists manage crop health more effectively.

## Features

- **Accurate Disease Detection**: Utilizes state-of-the-art machine learning models to identify plant diseases.
- **User-Friendly Interface**: Easy-to-use interface for uploading leaf images and receiving diagnostic results.
- **Comprehensive Database**: Extensive dataset of plant leaves with various diseases for training and validation.
- **Real-Time Processing**: Fast and efficient processing of images to provide real-time results.

## Technologies Used

- **Programming Languages**: JavaScript
- **Libraries**: TensorFlow.js, Node.js, Express, Multer
- **Frameworks**: Express for the web interface
- **Tools**: Jupyter Notebook for experimentation and visualization (if using Python for training)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/Plant-Leaf-Disease-Detection-System-Using-AI-Algorithms.git
    cd Plant-Leaf-Disease-Detection-System-Using-AI-Algorithms
    ```

2. **Create and activate a virtual environment (if needed for any Python preprocessing or training):**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    npm install
    ```

4. **Download the dataset:**

    Download the dataset from [Kaggle](https://www.kaggle.com) and place it in the `data` directory.

5. **Run the application:**

    ```bash
    node app.js
    ```

## Usage

1. **Upload Image**: Upload an image of a plant leaf through the web interface.
2. **Processing**: The system processes the image and runs it through the trained machine learning model.
3. **Results**: The system displays the disease detected in the leaf, if any, along with confidence scores.

## Dataset

The dataset used in this project contains images of healthy and diseased plant leaves. Each image is labeled with the type of disease. The dataset is split into training, validation, and test sets.

## Model Training

1. **Data Preprocessing**: Images are resized, normalized, and augmented to enhance model performance.
2. **Model Architecture**: A Convolutional Neural Network (CNN) is used for image classification.
3. **Training**: The model is trained on the processed dataset using TensorFlow/Keras.
4. **Evaluation**: Model performance is evaluated on the validation set and fine-tuned for optimal accuracy.

## Proposed Model

### Dataset
- Plant village dataset with tomato samples, including six disorders.

### Preprocessing
1. **Conversion to Grayscale**:
    - Grayscale images reduce complexity and enhance feature extraction.
2. **Histogram Equalization (HE)**:
    - Improves the contrast of the images.
3. **K-means Clustering**:
    - Segments the image into clusters to highlight the diseased regions.
4. **Contour Tracing**:
    - Detects the boundaries of diseased spots on the leaves.

### Feature Extraction
1. **Discrete Wavelet Transform (DWT)**:
    - Decomposes the image into frequency components.
2. **Principal Component Analysis (PCA)**:
    - Reduces the dimensionality of the data, retaining the most important features.
3. **Gray-Level Co-occurrence Matrix (GLCM)**:
    - Extracts texture features from the image.

### Classification
1. **Support Vector Machine (SVM)**:
    - Classifies the data by finding the hyperplane that best separates the classes.
2. **K-Nearest Neighbors (K-NN)**:
    - Classifies data points based on the majority class among the nearest neighbors.
3. **Convolutional Neural Network (CNN)**:
    - Deep learning model that automatically extracts features and classifies the images.

### Evaluation Metrics
- **Precision**:
    - The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**:
    - The ratio of correctly predicted positive observations to all the observations in the actual class.
- **F1 Score**:
    - The weighted average of Precision and Recall.

## Experimental Results
The proposed model was tested on a tomato leaf disease dataset with 600 samples. The results are as follows:
- **Healthy Leaf**: 99% accuracy.
- **Mosaic Virus**: 100% accuracy.
- **Leaf Mold**: 100% accuracy.
- **Yellow Curl**: 99% accuracy.
- **Spotted Spider Mite**: 99% accuracy.
- **Target Spot**: 100% accuracy.
- **Overall Accuracy**: 99.5%.

The proposed model, using DWT, PCA, GLCM, and CNN, provides better accuracy compared to existing models.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## Acknowledgements

- Thanks to [Kaggle](https://www.kaggle.com) for providing the dataset.
- Inspiration from various open-source AI projects.

## Contact

For any inquiries, please contact:

- Name: Sunhith Reddy
- Email: iamtsr2004@gmail.com
- Linkedin: www.linkedin.com/in/sunhith-reddy-t


