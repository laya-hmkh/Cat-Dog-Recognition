# Cat-Dog-Recognition

# Convolutional Neural Network for Image Classification

This repository contains a Convolutional Neural Network (CNN) built using TensorFlow and Keras for image classification. The model is designed to classify images into two categories (e.g., cats vs. dogs) by learning from a provided dataset.

## Project Overview

The project follows these main steps:

1.  **Data Preprocessing:**
    *   **Training Set:** Uses `ImageDataGenerator` for data augmentation (rescaling, shearing, zooming, horizontal flipping) to prevent overfitting. Images are loaded from a directory named `dataset\\training_set`, resized to 64x64 pixels and batched.
    *   **Test Set:** Similar to training set preprocessing but with only rescaling. Also reads images from the same directory.
2.  **Building the CNN:**
    *   **Initialization:** A sequential model is initialized.
    *   **Convolution:** Two `Conv2D` layers with ReLU activation are added, followed by max pooling to extract features.
    *   **Flattening:** The output from the convolutional layers is flattened into a 1D vector.
    *   **Full Connection:** A fully connected dense layer with ReLU activation is added.
    *   **Output Layer:** An output layer with a sigmoid activation is added for binary classification.
3.  **Training the CNN:**
    *   **Compilation:** The model is compiled using the Adam optimizer, binary cross-entropy loss, and accuracy metric.
    *   **Training:** The model is trained using the preprocessed training data and evaluated on the test data for 25 epochs.
4.  **Making a Single Prediction:**
    *   **Image Loading:** Loads and preprocesses a single test image from a specified directory, which the user should change to point to their specific image.
    *   **Prediction:** Uses the trained model to predict the class of the image.
    *   **Output:** Prints the predicted class ("Dog" or "Cat").

## Libraries Used

*   `tensorflow`
*   `keras`
*   `numpy`

## How to Use

1.  Ensure you have the necessary libraries installed (`tensorflow`, `keras`, `numpy`).
2.  Organize your training data into a directory structure like this:
    ```
    dataset/
        training_set/
            cat/
                cat1.jpg
                cat2.jpg
                ...
            dog/
                dog1.jpg
                dog2.jpg
                ...
    ```
3.  Modify the path `"Directory"` in the "Making a single Prediction" section to point to the location of your single image you wish to predict.
4.  Run the script.

## Notes

*   This example showcases a basic image classification model.
*   The directory `dataset\\training_set` will need to be replaced by your own training dataset.
*   You can extend this example with other architectures, optimizers, and preprocessing methods to improve accuracy.
