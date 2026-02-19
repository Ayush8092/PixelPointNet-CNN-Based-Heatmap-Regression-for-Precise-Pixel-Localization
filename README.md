# Pixel Localization Using Deep Learning

_Supervised Regression via Heatmap-Based CNN_

1. Problem Statement

The goal of this project is to predict the spatial location (x,y) of a single bright pixel (intensity 255) in a 50×50 grayscale image, where all other pixels have value 0.


Formally:

-Input: a 50×50 grayscale image.
-Constraint: exactly one pixel has value 255, all others are 0 (plus injected noise/blur in this project).
-Output: pixel coordinates (x, y) of the bright pixel.

This is a 2D localization / keypoint detection problem, analogous to:

-Object keypoint detection (e.g., human pose joints).
-Anatomical landmark localization in medical imaging.
-Pixel-level attention / fixation prediction.

The assignment constraints:

-Use Deep Learning to perform supervised regression.
-You may generate your own dataset and must explain the design choices.
-Provide training logs, visualizations, and comparison of ground truth vs predicted coordinates.

2. Approach Overview

Instead of directly regressing the (x, y) coordinates, the project uses a 'heatmap regression' approach.

## Two possible approaches

I. Direct Coordinate Regression:

-Model directly outputs normalized (x, y) values.
-Loss: MSE/MAE in coordinate space.

II. Heatmap Regression (Chosen):

-Model outputs a 50×50 probability heatmap.
-The peak of the heatmap corresponds to the predicted pixel location.
-Final (x, y) is obtained via argmax on the heatmap.

##. Why Heatmap Regression?

We choose heatmap regression because:

-It preserves spatial structure (fully convolutional).
-It provides dense supervision instead of a single point loss.
-It is more robust to noise and slight label shifts.
-It is standard in pose estimation and landmark detection.
-It offers better interpretability: you can visualize where the model believes the point is.

The model is therefore a fully convolutional CNN trained to predict a 2D Gaussian heatmap centered at the true pixel, and we recover the coordinates via the maximum response location.


3. Dataset Design:

I. Synthetic Dataset

No dataset was provided, so the project generates a synthetic dataset tailored to the task.

## Why synthetic data?

-Full control over exact pixel-level ground truth.
-Unlimited data generation to prevent overfitting.
-Zero annotation noise, unlike real datasets.
-Ability to systematically vary difficulty and noise to stress-test the model.

II. Image Generation Process

For each sample:

i.Initialize an empty 50×50 grayscale image.
ii.Sample a random integer coordinate (x,y) within the image bounds.
iii.Set img[y,x] = 255 (single bright pixel).
iv.Apply Gaussian blur using OpenCV to emulate sensor blur: 
      e.g., cv2.GaussianBlur(img, (5, 5), 0).
v.Add Gaussian noise to mimic sensor noise:
      e.g., np.random.normal(0, 10, img.shape).
vi.Add random low-intensity background noise: a small fraction of pixels receive a random value between 5 and 40.
vii.Clip to [0, 255] and normalize to [0, 1].


This produces images that still contain a dominant bright region around the target pixel, but with realistic imperfections.

III. Heatmap Label Generation:

Instead of a single 1-hot pixel, each ground-truth label is a 2D Gaussian heatmap centered at the true coordinate:

-Heatmap size: 50×50.
-Center: (x,y).
-Spread is controlled via a parameter sigma (e.g., 1.8), which determines how many neighboring pixels get non-zero values.

Implementation: a function gaussian_heatmap(size, center, sigma) returns a 2D numpy array where values smoothly decay from the center.

## Benefits of Gaussian heatmaps:

-Provide dense gradients: nearby pixels also contribute to the loss.
-Encourage the model to learn spatial smoothness.
-Still allow a crisp point estimate via argmax.

IV. Splitting the Dataset:

The dataset is split into:

-Train: 80% of samples.
-Test: 20% of samples.

Additionally, a validation split is created from the training set (10% of train) for hyperparameter tuning and model selection.

Example:

-train_test_split(..., test_size=0.2, random_state=42) for train/test.
-Another train_test_split on the train set for train/validation.

4. Model Architecture

The model is a fully convolutional CNN implemented with tensorflow.keras.

I. Network Structure:

The model-building function (e.g., build_heatmap_model()) consists of:

-Input: (50, 50, 1) grayscale image.

-Several Conv2D + ReLU + BatchNormalization blocks, such as:
 Conv2D(32, kernel_size=3, padding='same', activation='relu')
 Conv2D(64, kernel_size=3, padding='same', activation='relu')
 Conv2D(128, kernel_size=3, padding='same', activation='relu')
 Conv2D(64, kernel_size=3, padding='same', activation='relu')
 Conv2D(32, kernel_size=3, padding='same', activation='relu')
 
-Final Conv2D layer:
Conv2D(1, kernel_size=1, activation='sigmoid', padding='same') to produce the heatmap.

Key properties:

-No pooling layers: prevents loss of spatial resolution.
-Fully convolutional: output resolution matches input resolution (50×50).
-Sigmoid output: per-pixel probability in [0, 1].

II. Loss and Metrics:

-Loss: Mean Squared Error (MSE) between predicted and true heatmaps.
-Training metric: Mean Absolute Error (MAE) on heatmaps.

In addition, the project defines a coordinate-space evaluation by taking argmax over heatmaps and computing MAE/MSE/RMSE on coordinates.


5. Training Setup:


I. Hyperparameters

-Optimizer: Adam.
-Loss: MSE.
-Metric: MAE (heatmap domain).
-Batch size: 64.
-Epochs: 75.
-Seeds: np.random.seed(42) and tf.random.set_seed(42) for reproducibility.

Example training call:

history = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    epochs=75,
    batch_size=64
)

II. Training Behavior:

Typical behavior observed:
-Training MSE drops rapidly from an initial high value to the order of 1e-3.
-Validation MSE remains low and stable, indicating no severe overfitting.
-Training and validation MAE curves are smooth and decreasing.

You can visualize:
-Training vs validation loss over epochs.
-Training vs validation MAE.

These plots help confirm convergence and stability.

6. Coordinate Extraction & Evaluation

Although the model is trained on heatmaps, the assignment requires coordinate regression. The project recovers coordinates and evaluates them explicitly.

I. Extracting Coordinates from Heatmaps

Given a predicted heatmap H of shape (50, 50):

i. Compute the index of the maximum value: idx = np.argmax(H)
ii. Convert flat index to 2D coordinates: y_pred, x_pred = np.unravel_index(idx, (50, 50))


A helper function such as extract_coords_from_heatmap(hm) encapsulates this logic and returns (x_pred, y_pred).

II. Metrics in Coordinate Space

To judge localization quality, the project computes:
-MAE (Mean Absolute Error) in pixels.
-MSE (Mean Squared Error) in pixels².
-RMSE (Root Mean Squared Error) in pixels.

Example utility:

from sklearn.metrics import mean_absolute_error, mean_squared_error
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


These metrics are computed separately on:
-Train coordinates vs predicted coordinates.
-Validation coordinates vs predicted coordinates.
-Test coordinates vs predicted coordinates.

III. Example Results

Typical results on this setup:

i. Train: 
-Very low MAE (well below 1 pixel on average).
-Low RMSE, indicating tight localization.

ii. Validation:
-Similar MAE/RMSE to train, suggesting good generalization.

iii. Test:
-Slightly higher but still small MAE/RMSE, demonstrating robust performance on unseen data.

In addition, a “mean pixel error” (average Euclidean distance per sample) is computed, along with its standard deviation, to summarize localization quality and dispersion.

7. Visualizations

Several visual diagnostics are included in the notebook:

I. Sample Images & Heatmaps

-Display of input images with noise and blur.
-Ground-truth heatmaps for those images.
-Predicted heatmaps from the model.

These visuals help confirm that the model is peaking near the correct location and not distracted by background noise.

II. Error Distributions

Histograms of:

-Absolute error in x-coordinate: |x_pred − x_true|.
-Absolute error in y-coordinate: |y_pred − y_true|.
-Euclidean pixel error per sample.

The distributions are typically concentrated near zero, indicating strong localization performance and low systematic bias. Broader spreads would signal instability or uncertainty.


8. Implementation Details & Code Structure

The main notebook (e.g., DeepEdge_ML_Assignment-3.ipynb) is structured into logical sections:

I. Problem Understanding & Approach
i. Heatmap Label Generation
ii. Realistic Dataset Generation
iii. Dataset Creation & Splitting
iv. Heatmap Regression CNN Model
v. Model Training
vi. Coordinate Extraction & Metrics
vii. Error Distribution & Analysis

The code uses:
-Type hints for key utility functions.
-Clear docstrings.
-Reproducibility via random seeds.
-Standard scientific Python libraries.


9. How to Run This Project

I. Requirements:
-Python 3.8+
-NumPy
-Matplotlib
-TensorFlow
-scikit-learn
-OpenCV

II. Installation
pip install numpy matplotlib tensorflow scikit-learn opencv-python

III. Running the Notebook
jupyter notebook DeepEdge_ML_Assignment-3.ipynb


10. Design Choices & Justification

I. Why Not Direct (x, y) Regression?
-Ignores spatial structure.
-Sparse supervision.
-No interpretability.

II. Why Heatmap Regression?
-Mimics real-world keypoint detection.
-Encourages spatial learning.
-Allows uncertainty visualization.

III. Why Noise & Blur?
-Models realistic sensor conditions.
-Improves robustness.
-Creates challenging yet controlled training samples.


11. Summary

This project implements a supervised deep learning pipeline for pixel-level localization using a heatmap regression CNN trained on a synthetic but realistic dataset.

Key outcomes:

i. Fully convolutional model predicts 2D Gaussian-like heatmaps.
ii. Coordinates extracted using argmax.
iii. Very low MAE/RMSE across train, validation, and test sets.
iv. Extensive visual and quantitative analysis confirms robust localization behavior.

This pipeline is directly extensible to advanced keypoint detection, pose estimation, and landmark localization tasks.
