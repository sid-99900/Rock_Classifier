# Rock_Classifier
CNN model for classifying different type of rock/mineral samples

This repository contains the code to reproduce the results from our paper, "Deep Learning based classification of visually similar geological materials from Microscopic Images under Sand Occlusion: A Terrestrial Analogue for Planetary Surface Imaging"

Requirements
Python 3.8+
TensorFlow 2.x

1.  **To augment and create more images:**
    Run the `augmentation.py` script. This will create more number of samples and create a more diverse set from limited images.

2.  **To train the model:**
    Run the `CNN.py` script. This will train the model and save the final weights.

3. **Deciding HSV Values per sample:**
   For calibrating HSV values for a single sample, use sand.py code, it helps in deciding a suitable HSV parameter

4. **For classifying images into different conditions:**
   sand_calculator.py is to be used for classifying an image into one of the three categories, clean image, lightly occluded or heavily occluded

   
