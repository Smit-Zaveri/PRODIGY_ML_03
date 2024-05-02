# Image Classification using Support Vector Machine (SVM)

This Python script performs image classification using the Support Vector Machine (SVM) algorithm. It utilizes a dataset consisting of images of cats and dogs.

## Requirements
- Python 3.x
- Required Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `opencv-python`, `plotly`, `missingno`, `keras`, `skimage`

## How to Use
1. Ensure you have Python installed on your system.
2. Install the required libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python plotly missingno keras scikit-image
```
OR
```bash
pip install -r requirements.txt
```

3. Download the dataset containing images of cats and dogs.
4. Extract the dataset and place it in the appropriate directory.
5. Run the script.

## Description
- The script imports necessary libraries for image processing, data manipulation, and machine learning.
- It extracts and preprocesses the images of cats and dogs from the dataset.
- Images are resized to a fixed size and flattened into a 1D array.
- Principal Component Analysis (PCA) is applied to reduce the dimensionality of the feature space.
- The data is split into training and testing sets.
- A Support Vector Machine (SVM) classifier is trained on the training data.
- Predictions are made on the testing data, and accuracy is calculated.
- Finally, the trained model is used to classify new images of cats and dogs.

## Video
![Video](03.mp4)

## Output
The script generates several outputs:
1. Visualization of sample images of cats and dogs from the dataset.
2. Evaluation of the SVM model, including accuracy score.
3. Prediction of labels for new images of cats and dogs.
4. CSV file (`svm_test_predictions.csv`) containing the predicted labels for test images.

## Author
[Smit Zaveri](https://github.com/Smit-Zaveri)

## Acknowledgments
- The dataset used in this script is obtained from [\[source link\]](https://www.kaggle.com/code/smitzaveri/svm-dog-cat/input).
- Inspiration for this script is derived from [source link].

