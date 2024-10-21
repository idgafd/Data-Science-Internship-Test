# Sentinel-2 Image Matching: Computer Vision Task

## Project Overview

In this task, we develop and implement algorithms for matching satellite images from the **Sentinel-2** dataset, focusing on matching images from different seasons. The objective is to detect keypoints in images and compare them to find similarities, which can be useful in monitoring environmental changes over time.

This project covers two main approaches:
1. **Classical Approach (SIFT)**: Using traditional feature detection and matching algorithms like **SIFT** to detect keypoints and match satellite images.
2. **Deep Learning Approach (Siamese Neural Network)**: Training a **Siamese Neural Network** to match satellite images by learning their similarities.

### Project Files
1. **Classical Approach (SIFT)**:
   - `DS_Internship_Sentinel_SIFT.ipynb`: Implements the classical image matching using the SIFT algorithm.

2. **Deep Learning Approach**:
   - `DS_Internship_Sentinel_DL.ipynb`: The main notebook for building and training a Siamese Neural Network on the Sentinel-2 dataset.
   - `DS_Internship_Sentinel_dataset.ipynb`: Dataset preparation notebook for processing Sentinel-2 tiles and reconstructing full images for training.
   - `DS_Internship_Sentinel_demo.ipynb`: Demonstration notebook for inference and visualization of the results using the trained model.
   - `sentinel_inference_model.py`: Python script for loading the trained model and running inference.
   - `sentinel_train_model.py`: Python script for training the Siamese Neural Network model from scratch.

### Raw Data, Preprocessed Data, and Model Weights
- **Raw Data**: Download the raw Sentinel-2 data from [Google Drive](https://drive.google.com/drive/u/0/folders/15e91Yvyhp6z6TAraA6C5J5vh0olRkx3D).
- **Preprocessed Dataset**: Preprocessed images ready for training can be found [here](https://drive.google.com/drive/u/0/folders/18PBdTIwK9R-cbddHWE_0whKZjajkCUCp).
- **Model Weights**: Pre-trained model weights are available [here](https://drive.google.com/file/d/1ECa6D4VvQaIl7bZv6A2HnZ4QK8uM1uqa/view?usp=drive_link).

---

## Instructions to Run the Project

### Step 1: Download Data and Model Weights
1. **Download Preprocessed Dataset**: Download the preprocessed dataset to use directly in the deep learning models.
2. **Download Model Weights**: If you want to skip the training step and directly perform inference, download the pre-trained model weights.

### Step 2: Correct Paths in the Code
After downloading the data and weights:
1. Update the paths in the notebooks and scripts to point to the correct directories for the raw/preprocessed data and model weights.
   - For example, in `sentinel_train_model.py` and `sentinel_inference_model.py`, update the paths to where you saved the data and weights.
   - Paths to update:
     ```python
     root_dir = '/path_to_your_data/Sentinel_dev_data'  # Update with your data location
     model_path = '/path_to_model_weights/siamese_model.pth'  # Update with your weights location
     ```

### Step 3: Run the Classical Approach (SIFT)
- Open `DS_Internship_Sentinel_SIFT.ipynb` and execute the cells.
- This notebook implements a classical image matching algorithm using the SIFT feature detector.

### Step 4: Train and Evaluate the Deep Learning Model
- To **train** the Siamese Neural Network, you can either:
   1. Run the training notebook `DS_Internship_Sentinel_DL.ipynb`, or
   2. Use the `sentinel_train_model.py` script:
     ```bash
     python sentinel_train_model.py
     ```
- To **perform inference** using the trained model, you can either:
   1. Run the demo notebook `DS_Internship_Sentinel_demo.ipynb`, or
   2. Use the `sentinel_inference_model.py` script:
     ```bash
     python sentinel_inference_model.py
     ```

### Step 5: Visualize Results
- In the demo notebook `DS_Internship_Sentinel_demo.ipynb`, you can visualize the keypoints and similarity scores of image pairs after performing inference.


