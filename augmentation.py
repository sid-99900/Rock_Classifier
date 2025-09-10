import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image, ImageOps
import numpy as np

source_dir = "enter/your/path"
save_dir = "/enter/your/path"
os.makedirs(save_dir, exist_ok=True)

# Number of augmented images per original image
AUGMENT_COUNT = 50

#augmentation settings
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=127 #gray fill to avoid stripe artifacts
)

# Loop over all images
for filename in os.listdir(source_dir):
    if not filename.lower().endswith(('jpg', 'jpeg', 'png')):
        continue 

    img_path = os.path.join(source_dir, filename)

    try:
        img = Image.open(img_path)
        pad_pixels = int(0.1 * max(img.size))
        padded_img = ImageOps.expand(img, border=pad_pixels, fill=(127, 127, 127))
        x = img_to_array(padded_img)
        x = x.reshape((1,) + x.shape)

    except Exception as e:
        print(f"Skipping {filename}: {e}")
        continue

    #Augment
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir,
                              save_prefix='aug', save_format='jpg'):
        i += 1
        if i >= AUGMENT_COUNT:
            break

print("Augmentation complete")

