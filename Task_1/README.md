# Mountain Named Entity Recognition (NER)

## Project Overview

This project focuses on building and training a Named Entity Recognition (NER) model to identify **mountain names** within textual data. The goal is to automatically recognize and label mountain names from a given dataset of sentences. We implement the NER task using state-of-the-art deep learning techniques, providing both training and inference scripts.

This project involves:
1. **Dataset Preparation**: Preparing a dataset with labeled mountain names.
2. **Model Architecture**: Selecting and training the NER model architecture.
3. **Inference**: Implementing a demo for running inference using the trained model.

### Project Files
1. **Dataset Preparation**:
   - `DS_Internship_NER_dataset.ipynb`: This notebook prepares the dataset for training and processes the CSV dataset containing sentences labeled with mountain names.
   
2. **Model Training and Inference**:
   - `DS_Internship_NER_model.ipynb`: The main notebook that handles model architecture, training, and evaluation.
   - `ner_train_model.py`: Python script to train the NER model from scratch.
   - `ner_inference_model.py`: Python script for loading the trained model and running inference on new sentences.
   - `DS_Internship_NER_demo.ipynb`: A notebook demonstrating how to use the trained model for inference and visualize the results.

### Resources
- **Label Encoder**: The label encoder used in the project is available [here](https://drive.google.com/file/d/1nF_4anhyZxyFIDcyJa-PNUObbazV1p_O/view?usp=drive_link).
- **Model Weights**: Pre-trained model weights for direct inference can be downloaded from [here](https://drive.google.com/file/d/17mY81zkwB_RXkH9RHUJ64t0_9Ax2doAH/view?usp=drive_link).
- **Dataset**: The dataset with labeled mountains can be downloaded from [here](https://drive.google.com/file/d/1prb4uItElSvRB1kn_eBUmJSkGhMIzmjT/view?usp=drive_link).

---

## Instructions to Run the Project

### Step 1: Download Data and Model Weights
1. **Download the dataset**: Download the labeled dataset of mountain names from the provided link.
2. **Download Label Encoder**: Download the label encoder used for the NER task.
3. **Download Model Weights**: If you want to skip the training step, download the pre-trained model weights for direct inference.

### Step 2: Set Up Paths in the Code
After downloading the data and weights:
1. Update the paths in the notebooks and scripts to point to the correct directories for the dataset, label encoder, and model weights.
   - For example, in `ner_train_model.py` and `ner_inference_model.py`, update the paths to where you saved the files.
   - Paths to update:
     ```python
     dataset_path = '/path_to_your_data/ner_dataset.csv'  # Update with your dataset location
     model_path = '/path_to_model_weights/ner_model.pth'  # Update with your weights location
     label_encoder_path = '/path_to_label_encoder/label_encoder.pkl'  # Update with your label encoder location
     ```

### Step 3: Train the NER Model
- To **train** the model, you can either:
   1. Run the training notebook `DS_Internship_NER_model.ipynb`, or
   2. Use the `ner_train_model.py` script:
     ```bash
     python ner_train_model.py
     ```
   This script will train the NER model on the dataset and save the trained model weights.

### Step 4: Perform Inference
- To **run inference** using the trained model, you can either:
   1. Run the demo notebook `DS_Internship_NER_demo.ipynb`, or
   2. Use the `ner_inference_model.py` script to predict mountain names in new text:
     ```bash
     python ner_inference_model.py
     ```

### Step 5: Visualize Results
- In the `DS_Internship_NER_demo.ipynb` notebook, you can visualize the results of the NER model. The notebook demonstrates how the model labels mountain names within the input text and provides visualizations for the predictions.

