# Pixel Localization Using Deep Learning 

_Supervised Regression via Heatmap-Based CNN_


## 1. Problem Statement

The goal of this project is to predict the spatial location \((x, y)\) of a single bright pixel (intensity 255) in a 50×50 grayscale image, where all other pixels have value 0.

Formally:

- Input: a 50×50 grayscale image.
- Constraint: exactly one pixel has value 255, all others are 0 (plus injected noise/blur in this project).
- Output: pixel coordinates (x, y) of the bright pixel.

This is a 2D localization / keypoint detection problem, analogous to:

- Object keypoint detection (e.g., human pose joints).
- Anatomical landmark localization in medical imaging.
- Pixel-level attention / fixation prediction.

The assignment constraints:

- Use Deep Learning to perform supervised regression.
- You may generate your own dataset and must explain the design choices.
- Provide training logs, visualizations, and comparison of ground truth vs predicted coordinates.



## 2. Approach Overview

Instead of directly regressing the (x, y) coordinates, the project uses a **heatmap regression** approach.

### Two possible approaches

1. **Direct Coordinate Regression**
   - Model directly outputs normalized (x, y) values.
   - Loss: MSE/MAE in coordinate space.

2. **Heatmap Regression (Chosen)**
   - Model outputs a 50×50 probability heatmap.
   - The peak of the heatmap corresponds to the predicted pixel location.
   - Final (x, y) is obtained via `argmax` on the heatmap.

### Why Heatmap Regression?

We choose heatmap regression because:

- It preserves spatial structure (fully convolutional).
- It provides dense supervision instead of a single point loss.
- It is more robust to noise and slight label shifts.
- It is standard in pose estimation and landmark detection.
- It offers better interpretability: you can visualize where the model believes the point is.

The model is therefore a fully convolutional CNN trained to predict a 2D Gaussian heatmap centered at the true pixel, and we recover the coordinates via the maximum response location.

---

## 3. Dataset Design

### 3.1 Synthetic Dataset

No dataset was provided, so the project generates a synthetic dataset tailored to the task.

**Why synthetic data?**

- Full control over exact pixel-level ground truth.
- Unlimited data generation to prevent overfitting.
- Zero annotation noise, unlike real datasets.
- Ability to systematically vary difficulty and noise to stress-test the model.

### 3.2 Image Generation Process

For each sample:

1. Initialize an empty 50×50 grayscale image.
2. Sample a random integer coordinate \((x, y)\) within the image bounds.
3. Set `img[y, x] = 255` (single bright pixel).
4. Apply Gaussian blur using OpenCV to emulate sensor blur:
   - e.g., `cv2.GaussianBlur(img, (5, 5), 0)`.
5. Add Gaussian noise to mimic sensor noise:
   - e.g., `np.random.normal(0, 10, img.shape)`.
6. Add random low-intensity background noise:
   - a small fraction of pixels receive a random value between 5 and 40.
7. Clip to [0, 255] and normalize to [0, 1].

This produces images that still contain a dominant bright region around the target pixel, but with realistic imperfections.

### 3.3 Heatmap Label Generation

Instead of a single 1-hot pixel, each ground-truth label is a 2D Gaussian heatmap centered at the true coordinate:

- Heatmap size: 50×50.
- Center: \((x, y)\).
- Spread is controlled via a parameter `sigma` (e.g., 1.8), which determines how many neighboring pixels get non-zero values.

Implementation: a function `gaussian_heatmap(size, center, sigma)` returns a 2D numpy array where values smoothly decay from the center.

**Benefits of Gaussian heatmaps:**

- Provide dense gradients: nearby pixels also contribute to the loss.
- Encourage the model to learn spatial smoothness.
- Still allow a crisp point estimate via `argmax`.

### 3.4 Splitting the Dataset

The dataset is split into:

- Train: 80% of samples.
- Test: 20% of samples.

Additionally, a validation split is created from the training set (10% of train) for hyperparameter tuning and model selection.

Example:

- `train_test_split(..., test_size=0.2, random_state=42)` for train/test.
- Another `train_test_split` on the train set for train/validation.

---

## 4. Model Architecture

The model is a fully convolutional CNN implemented with `tensorflow.keras`.

### 4.1 Network Structure

The model-building function (e.g., `build_heatmap_model()`) consists of:

- Input: `(50, 50, 1)` grayscale image.
- Several Conv2D + ReLU + BatchNormalization blocks, such as:
  - Conv2D(32, kernel_size=3, padding='same', activation='relu')
  - Conv2D(64, kernel_size=3, padding='same', activation='relu')
  - Conv2D(128, kernel_size=3, padding='same', activation='relu')
  - Conv2D(64, kernel_size=3, padding='same', activation='relu')
  - Conv2D(32, kernel_size=3, padding='same', activation='relu')
- Final Conv2D layer:
  - Conv2D(1, kernel_size=1, activation='sigmoid', padding='same') to produce the heatmap.

Key properties:

- No pooling layers: prevents loss of spatial resolution.
- Fully convolutional: output resolution matches input resolution (50×50).
- Sigmoid output: per-pixel probability in [0, 1].

### 4.2 Loss and Metrics

- Loss: Mean Squared Error (MSE) between predicted and true heatmaps.
- Training metric: Mean Absolute Error (MAE) on heatmaps.

In addition, the project defines a coordinate-space evaluation (see next section) by taking argmax over heatmaps and computing MAE/MSE/RMSE on coordinates.

---

## 5. Training Setup

### 5.1 Hyperparameters

- Optimizer: Adam.
- Loss: MSE.
- Metric: MAE (heatmap domain).
- Batch size: 64.
- Epochs: 75.
- Seeds: `np.random.seed(42)` and `tf.random.set_seed(42)` for reproducibility.

Example training call:

```python
history = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    epochs=75,
    batch_size=64
)

5.2 Training Behavior

Typical behavior observed:

    Training MSE drops rapidly from an initial high value to the order of 1e-3.

    Validation MSE remains low and stable, indicating no severe overfitting.

    Training and validation MAE curves are smooth and decreasing.

You can visualize:

    Training vs validation loss over epochs.

    Training vs validation MAE.

These plots help confirm convergence and stability.
6. Coordinate Extraction & Evaluation

Although the model is trained on heatmaps, the assignment requires coordinate regression. The project recovers coordinates and evaluates them explicitly.
6.1 Extracting Coordinates from Heatmaps

Given a predicted heatmap H of shape (50, 50):

    Compute the index of the maximum value:

    python
    idx = np.argmax(H)

    Convert flat index to 2D coordinates:

    python
    y_pred, x_pred = np.unravel_index(idx, (50, 50))

A helper function such as extract_coords_from_heatmap(hm) encapsulates this logic and returns (x_pred, y_pred).
6.2 Metrics in Coordinate Space

To judge localization quality, the project computes:

    MAE (Mean Absolute Error) in pixels.

    MSE (Mean Squared Error) in pixels².

    RMSE (Root Mean Squared Error) in pixels.

Example utility:

python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

These metrics are computed separately on:

    Train coordinates vs predicted coordinates.

    Validation coordinates vs predicted coordinates.

    Test coordinates vs predicted coordinates.

6.3 Example Results

Typical results on this setup:

    Train:

        Very low MAE (well below 1 pixel on average).

        Low RMSE, indicating tight localization.

    Validation:

        Similar MAE/RMSE to train, suggesting good generalization.

    Test:

        Slightly higher but still small MAE/RMSE, demonstrating robust performance on unseen data.

In addition, a “mean pixel error” (average Euclidean distance per sample) is computed, along with its standard deviation, to summarize localization quality and dispersion.
7. Visualizations

Several visual diagnostics are included in the notebook:
7.1 Sample Images & Heatmaps

    Display of input images with noise and blur.

    Ground-truth heatmaps for those images.

    Predicted heatmaps from the model.

These visuals help confirm that the model is peaking near the correct location and not distracted by background noise.
7.2 Error Distributions

Histograms of:

    Absolute error in x-coordinate: |x_pred - x_true|.

    Absolute error in y-coordinate: |y_pred - y_true|.

    Euclidean pixel error per sample.

The distributions are typically concentrated near zero, indicating strong localization performance and low systematic bias. Broader spreads would signal instability or uncertainty.

These plots are important to understand not just average error but full error behavior.
8. Implementation Details & Code Structure

The main notebook (e.g., DeepEdge_ML_Assignment-3.ipynb) is structured into logical sections:

    Problem Understanding & Approach

        Problem statement, use-cases, and high-level plan.

    Heatmap Label Generation

        gaussian_heatmap function and rationale.

    Realistic Dataset Generation

        generate_dataset with noise, blur, and background artifacts.

    Dataset Creation & Splitting

        Train/test split, and further train/validation split.

    Heatmap Regression CNN Model

        build_heatmap_model fully-convolutional architecture.

    Model Training

        Training call with history tracking.

    Coordinate Extraction & Metrics

        extract_coords_from_heatmap, compute_metrics, and MAE/MSE/RMSE on train/validation/test.

    Error Distribution & Analysis

        Histograms for x/y and Euclidean errors, plus narrative analysis.

The code uses:

    Type hints for key utility functions.

    Clear docstrings (purpose, arguments, return values).

    Reproducibility via random seeds.

    Standard libraries (numpy, tensorflow.keras, matplotlib, sklearn, cv2).

9. How to Run This Project
9.1 Requirements

Core dependencies:

    Python 3.8+

    NumPy

    Matplotlib

    TensorFlow (with Keras)

    scikit-learn

    OpenCV (cv2)

9.2 Installation

Using pip:

bash
pip install numpy matplotlib tensorflow scikit-learn opencv-python

For GPU acceleration, install the appropriate TensorFlow package for your CUDA/CuDNN setup.
9.3 Running the Notebook

    Clone the repository:

    bash
    git clone https://github.com/<your-username>/deepedge-pixel-localization.git
    cd deepedge-pixel-localization

    Open the notebook:

    bash
    jupyter notebook DeepEdge_ML_Assignment-3.ipynb

    or upload the .ipynb file to Google Colab.

    Run all cells sequentially.

This will:

    Generate the synthetic dataset.

    Build and train the heatmap regression CNN.

    Extract coordinates from predicted heatmaps.

    Compute train/validation/test MAE/MSE/RMSE in pixel space.

    Plot error distributions and visualizations.

10. Design Choices & Justification
10.1 Why Not Direct (x, y) Regression?

Directly regressing coordinates would work for this simple setup, but:

    It ignores image-wide spatial structure.

    It provides sparse supervision (only two target values per sample).

    It is less interpretable: you cannot see where the model is uncertain.

10.2 Why Heatmap Regression?

    Mimics real-world keypoint detection setups.

    Encourages the model to learn spatial patterns around the bright pixel.

    Allows inspection of the full belief map, not just the final argmax.

10.3 Why Noise & Blur?

    Real camera sensors produce blurred and noisy images.

    Adding Gaussian noise, blur, and background clutter increases robustness.

    Serves as a controlled benchmark where the model must still localize correctly despite distortions.

11. Summary

This project implements a supervised deep learning pipeline for pixel-level localization using a heatmap regression CNN trained on a synthetic but realistic dataset.

Key outcomes:

    Fully convolutional model predicts a 2D Gaussian-like heatmap per image.

    Coordinates are extracted via argmax on the heatmap.

    The model achieves very low MAE/RMSE on train, validation, and test sets, even with blur and noise.

    Extensive visual and quantitative analysis confirms robust localization behavior.

The approach is directly extensible to more complex keypoint detection and landmark localization tasks in real-world computer vision applications.
