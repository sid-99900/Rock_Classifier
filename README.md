# Rock_Classifier
CNN model for classifying different type of rock/mineral samples

This repository contains the code to reproduce the results from our paper, "Deep Learning based classification of visually similar geological materials from Microscopic Images under Sand Occlusion: A Terrestrial Analogue for Planetary Surface Imaging"

Requirements on a local computer
Python 3.8+
TensorFlow 2.x

1.  **To train the model and analyze the results:**
    Run the `CNN.py` script. This is for end-to-end model training, cross-validation and evaluation pipeline for the 5-class rock classification task (basalt,
    magnetite, calcite, quartz, graphite).Key features handled in this script include: Automated Extraction & Balancing: Unzips dataset.zip, standardizes counts
    across the 3 imaging conditions (clean, light, heavy), and performs a proportionate 70/10/20 train/val/test split. Condition-Aware Data Augmentation: Augments
    each class-condition pair up to a target size of 400 images, yielding a total training set of 6,000 images across 5 classes. Model Architecture: Fine-tunes an
    InceptionV3 backbone (pre-trained on ImageNet) using Transfer Learning with Early Stopping and Model Checkpointing.Statistical Validation ($K$-Fold Cross
    Validation): Conducts a 3-Fold Cross-Validation loop across the dataset to measure generalized out-of-fold metrics (Accuracy, Precision, Recall,
    F1).Comprehensive Metric Export: Generates and exports publication-ready high-DPI figures (Training curves, Confusion Matrix, One-vs-Rest ROC/AUC curves,
    Condition-wise Accuracy bar charts) and consolidated spreadsheets (all_results_summary.xlsx).

2. **Deciding HSV Values per sample:**
   For calibrating HSV values for a single sample, use sand.py code, it helps in deciding a suitable HSV parameter

3. **For classifying images into different conditions:**
   sand_calculator.py is to be used for classifying an image into one of the three categories, clean image, lightly occluded or heavily occluded

   
