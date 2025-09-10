# Rock_Classifier
CNN model for classifying different type of rock samples

This repository contains the code to reproduce the results from our paper, "Deep Learning-Based Classification of Magnetite and Basalt from USB Microscope Imagery for Planetary Analog Studies"

Requirements
Python 3.8+
TensorFlow 2.x

1.  **To augment and create more images**
    Run the `augmentation.py` script. This will create more number of samples and create a more diverse set from limited images.

1.  **To train the model:**
    Run the `CNN.py` script. This will train the model and save the final weights.

2.  **To run the 3-fold cross-validation:**
    Run the `cross_validation.py` script. This will print the validation accuracy for each fold.
