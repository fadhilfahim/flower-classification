# Flower Classification

## Project Overview

This project involves building a Convolutional Neural Network (CNN) for classifying images of flowers. The dataset consists of images from various flower species, and the goal is to train a model that can accurately predict the species of a flower from a given image. The model uses TensorFlow and Keras to implement deep learning techniques for image classification.

## Key Steps in the Process

### 1. **Data Preprocessing**
   - **Directory Setup**: The dataset is organized into subdirectories, each representing a different flower class. The images are stored in these subdirectories, with each folder name corresponding to the class label.
   - **Image Validation**: The code checks if the images are valid by attempting to open and verify each image. Invalid or corrupted files are removed from the dataset.
   - **File Removal**: Non-image files (e.g., text files) are removed from the dataset to ensure only valid image files remain for training.

### 2. **Dataset Loading**
   - **Image Loading**: The dataset is loaded using TensorFlow's `image_dataset_from_directory` function, which automatically labels images based on their directory structure. The images are resized to a uniform size of 180x180 pixels.
   - **Training and Validation Split**: The dataset is split into training and validation sets using a 80/20 split, with the validation set used to evaluate the model's performance during training.

### 3. **Data Augmentation**
   - **Augmentation Techniques**: The model uses data augmentation techniques such as horizontal flipping, random rotation, and random zoom to artificially increase the size of the training dataset and make the model more robust.
   - **Visualization**: A few augmented images are visualized to confirm the augmentation process.

### 4. **Model Creation**
   - **Sequential Model**: A CNN model is built using Keras' Sequential API. The model consists of several layers:
     - **Data Augmentation Layer**: Applied to the images before they are fed into the model.
     - **Convolutional Layers**: These layers apply convolution operations to the input images to extract important features.
     - **MaxPooling Layers**: These layers downsample the feature maps, reducing the spatial dimensions and computational load.
     - **Dropout Layer**: This layer helps prevent overfitting by randomly setting a fraction of input units to zero during training.
     - **Flatten Layer**: Converts the 2D feature maps into 1D vectors for the fully connected layers.
     - **Dense Layers**: Fully connected layers that make predictions based on the extracted features.
   - **Rescaling**: The pixel values of the images are scaled between 0 and 1 to normalize the data.
   - **Output Layer**: The final layer uses a softmax activation function to output probabilities for each flower class.

### 5. **Model Compilation**
   - The model is compiled with the Adam optimizer and Sparse Categorical Crossentropy loss function, which is suitable for multi-class classification tasks.
   - **Metrics**: The model is evaluated using accuracy as the metric.

### 6. **Model Training**
   - The model is trained for 15 epochs using the training dataset, with validation performed on the test dataset after each epoch. The training and validation accuracy are monitored to assess the model's performance.

### 7. **Prediction**
   - A function `classify_image` is defined to make predictions on new images. The function loads the image, preprocesses it, and feeds it into the trained model to predict the flower class.
   - The model outputs the predicted class along with the prediction confidence.

### 8. **Model Saving**
   - After training, the model is saved to a file (`flower_classification_model.keras`) for future use or deployment.

## Libraries and Tools Used
- **TensorFlow**: A deep learning framework used to build and train the CNN model.
- **Keras**: A high-level API for building neural networks, which is part of TensorFlow.
- **NumPy**: A library for numerical operations, used for handling arrays and mathematical operations.
- **PIL (Python Imaging Library)**: Used for opening, manipulating, and verifying image files.
- **Matplotlib**: A plotting library used for visualizing images and training results.
- **OS**: The operating system library is used to navigate the directory structure and remove invalid files.

## Model Details
- **Model Architecture**: Convolutional Neural Network (CNN) with data augmentation, convolutional layers, pooling layers, dropout, and dense layers.
- **Input Size**: The images are resized to 180x180 pixels before being fed into the model.
- **Output**: The model predicts one of five flower classes based on the input image.

## How to Use
1. **Train the Model**: Run the script to train the model using the flower dataset.
2. **Make Predictions**: Use the `classify_image` function to classify new flower images by passing the image path.
3. **Save and Load Model**: The trained model can be saved and loaded using `model.save()` and `load_model()` for later use.

---

This summary should help users understand the steps involved in the project and how the model is built and used.