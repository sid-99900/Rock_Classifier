import os
import random
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

# 1. SETUP PATHS
source_root = r"C:\Users\Siddhant\Desktop\dataset"
output_root = r"C:\Users\Siddhant\Desktop\BTP_Final_Dataset"
materials = ['basalt', 'magnetite', 'graphite', 'calcite', 'quartz']

# 2. AUGMENTATION SETTINGS [cite: 58-62]
datagen = ImageDataGenerator(
    rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
    zoom_range=0.3, shear_range=0.2, brightness_range=[0.7, 1.3],
    horizontal_flip=True, vertical_flip=True, fill_mode='constant', 
    cval=127 # Gray fill for edge artifacts
)

CLEAN_TARGET = 2500
SAND_TOTAL_TARGET = 2500 
TEST_SPLIT = 0.2

# 3. DIRECTORY SETUP
for m in materials:
    for s in ['train', 'test']:
        os.makedirs(os.path.join(output_root, s, m), exist_ok=True)

# 4. RAM-EFFICIENT PROCESSING
for mat in materials:
    print(f"--- Processing {mat} ---")
    mat_path = os.path.join(source_root, mat)
    subs = os.listdir(mat_path)
    
    # Identify subfolders correctly
    c_sub = [f for f in subs if f == mat][0]
    l_sub = [f for f in subs if 'light' in f][0]
    h_sub = [f for f in subs if ('sand' in f or 'heavy' in f) and 'light' not in f][0]

    tasks = [(c_sub, CLEAN_TARGET, 'clean'), 
             (l_sub, 1250, 'light'), 
             (h_sub, 1250, 'heavy')]

    for sub_name, target, tag in tasks:
        sub_p = os.path.join(mat_path, sub_name)
        files = [f for f in os.listdir(sub_p) if f.lower().endswith(('.jpg', '.png'))]
        
        count = 0
        while count < target:
            for fname in files:
                if count >= target: break
                img = Image.open(os.path.join(sub_p, fname)).convert('RGB')
                
                # Standardize with 10% gray border
                pad = int(0.1 * max(img.size))
                img = ImageOps.expand(img, border=pad, fill=(127, 127, 127))
                
                # Augment only if we've already used the original images
                if count >= len(files):
                    x = img_to_array(img).reshape((1,) + (img.height, img.width, 3))
                    for batch in datagen.flow(x, batch_size=1):
                        img = array_to_img(batch[0])
                        break
                
                # Determine split and save immediately to SSD
                folder = 'test' if random.random() < TEST_SPLIT else 'train'
                save_p = os.path.join(output_root, folder, mat, f"{mat}_{tag}_{count}.jpg")
                img.save(save_p)
                count += 1

print(f"✅ Balanced dataset generated at {output_root}")
