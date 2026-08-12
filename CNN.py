"""
=============================================================
REVISED CNN TRAINING PIPELINE
Proportionate augmentation per class per condition
Target: equal images per condition per class
=============================================================
"""

# ─────────────────────────────────────────────────────────────
# CELL 1 — Upload & Unzip
# ─────────────────────────────────────────────────────────────
# %%
import os, shutil, glob, random, math, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array, array_to_img)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

ZIP_FILENAME = "dataset.zip"
ZIP_PATH     = f"/content/{ZIP_FILENAME}"
EXTRACT_DIR  = "/content/extracted"
RESULTS_DIR  = "/content/CG_Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(ZIP_PATH):
    print(f"ERROR: Upload {ZIP_FILENAME} first via the Files panel.")
else:
    print(f"Found {ZIP_PATH} — {os.path.getsize(ZIP_PATH)/1024/1024:.1f} MB")
    if not os.path.exists(EXTRACT_DIR):
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(EXTRACT_DIR)
        print("Done.")
    else:
        print("Already extracted.")

    contents = os.listdir(EXTRACT_DIR)
    ORIGINAL_DATA_DIR = (
        os.path.join(EXTRACT_DIR, contents[0])
        if len(contents) == 1 and os.path.isdir(
            os.path.join(EXTRACT_DIR, contents[0]))
        else EXTRACT_DIR
    )
    print(f"Dataset root: {ORIGINAL_DATA_DIR}")
    print(f"Contents    : {os.listdir(ORIGINAL_DATA_DIR)}")


# ─────────────────────────────────────────────────────────────
# CELL 2 — Configuration
# ─────────────────────────────────────────────────────────────
# %%
CLASSES = ["basalt", "magnetite", "calcite", "quartz", "graphite"]

# Subfolder names inside each class folder
# Format: class → {condition: subfolder_name}
CLASS_SUBFOLDERS = {
    "basalt"    : {"clean": "basalt",
                   "light": "basalt_light_sand",
                   "heavy": "basalt_sand"},
    "magnetite" : {"clean": "magnetite",
                   "light": "magnetite_light_sand",
                   "heavy": "magnetite_sand"},
    "calcite"   : {"clean": "calcite",
                   "light": "calcite_light_sand",
                   "heavy": "calcite_sand"},
    "quartz"    : {"clean": "quartz",
                   "light": "quartz_light_sand",
                   "heavy": "quartz_sand"},
    "graphite"  : {"clean": "graphite",
                   "light": "graphite_light_sand",
                   "heavy": "graphite_sand"},
}
CONDITIONS = ["clean", "light", "heavy"]

# ── ACTUAL IMAGE COUNTS (update if you add more images) ───────
ACTUAL_COUNTS = {
    "basalt"    : {"clean": 72,  "light": 57,  "heavy": 64},
    "magnetite" : {"clean": 74,  "light": 71,  "heavy": 56},
    "calcite"   : {"clean": 73,  "light": 67,  "heavy": 55},
    "quartz"    : {"clean": 78,  "light": 67,  "heavy": 93},
    "graphite"  : {"clean": 74,  "light": 73,  "heavy": 81},
}

# ── PROPORTIONATE TARGET CALCULATION ─────────────────────────
# Find minimum count across ALL classes and ALL conditions
# This ensures every condition contributes equally
all_counts = [
    ACTUAL_COUNTS[cls][cond]
    for cls in CLASSES
    for cond in CONDITIONS
]
MIN_PER_CONDITION = min(all_counts)  # = 55 (calcite heavy)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

# Originals per condition after split
ORIG_TRAIN = max(1, int(MIN_PER_CONDITION * TRAIN_RATIO))  # ~38
ORIG_VAL   = max(1, int(MIN_PER_CONDITION * VAL_RATIO))    # ~5
ORIG_TEST  = max(1, int(MIN_PER_CONDITION * TEST_RATIO))   # ~11

# Augmentation target per condition in training
# Augment each condition to 400 images → 1200 per class total
AUG_TARGET_PER_CONDITION = 400

IMG_SIZE      = (299, 299)
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 0.0001
NUM_FOLDS     = 3
RANDOM_SEED   = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print(f"Minimum images across all conditions: {MIN_PER_CONDITION}")
print(f"Originals per condition: "
      f"{ORIG_TRAIN} train / {ORIG_VAL} val / {ORIG_TEST} test")
print(f"After augmentation per condition: {AUG_TARGET_PER_CONDITION}")
print(f"Total training per class: "
      f"{AUG_TARGET_PER_CONDITION * len(CONDITIONS)} "
      f"({AUG_TARGET_PER_CONDITION} × {len(CONDITIONS)} conditions)")
print(f"Total training all classes: "
      f"{AUG_TARGET_PER_CONDITION * len(CONDITIONS) * len(CLASSES)}")


# ─────────────────────────────────────────────────────────────
# CELL 3 — Verify Image Counts
# ─────────────────────────────────────────────────────────────
# %%
print("Verifying image counts from disk...\n")

for cls in CLASSES:
    cls_dir = os.path.join(ORIGINAL_DATA_DIR, cls)
    for cond in CONDITIONS:
        subfolder = CLASS_SUBFOLDERS[cls][cond]
        path = os.path.join(cls_dir, subfolder)
        if os.path.exists(path):
            imgs = [f for f in os.listdir(path)
                    if f.lower().endswith(
                        ('.jpg','.jpeg','.png','.bmp','.tiff'))]
            expected = ACTUAL_COUNTS[cls][cond]
            status = "✓" if len(imgs) >= expected else "⚠ LOW"
            print(f"  {cls:12s} {cond:6s}: "
                  f"{len(imgs):3d} found  "
                  f"(expected {expected})  {status}")
        else:
            print(f"  {cls:12s} {cond:6s}: PATH NOT FOUND — {path}")


# ─────────────────────────────────────────────────────────────
# CELL 4 — Split Originals (proportionate, condition-aware)
# ─────────────────────────────────────────────────────────────
# %%
SPLIT_DIR = "/content/split_originals"

def split_proportionate(original_dir, split_dir, seed=42):
    """
    For each class and each condition:
    - Cap images at MIN_PER_CONDITION to equalise
    - Split capped set into train/val/test
    - Rename files with cls_condition_ prefix for tracking
    """
    np.random.seed(seed)

    for cls in CLASSES:
        cls_dir = os.path.join(original_dir, cls)

        for cond in CONDITIONS:
            subfolder = CLASS_SUBFOLDERS[cls][cond]
            src_path  = os.path.join(cls_dir, subfolder)

            if not os.path.exists(src_path):
                print(f"  SKIP: {src_path} not found")
                continue

            imgs = [f for f in os.listdir(src_path)
                    if f.lower().endswith(
                        ('.jpg','.jpeg','.png','.bmp','.tiff'))]
            np.random.shuffle(imgs)

            # Cap to MIN_PER_CONDITION for balance
            imgs = imgs[:MIN_PER_CONDITION]
            n    = len(imgs)

            n_test  = max(1, int(n * TEST_RATIO))
            n_val   = max(1, int(n * VAL_RATIO))
            n_train = n - n_test - n_val

            splits = {
                "train" : imgs[:n_train],
                "val"   : imgs[n_train : n_train + n_val],
                "test"  : imgs[n_train + n_val :]
            }

            for split_name, split_imgs in splits.items():
                dest = os.path.join(split_dir, split_name, cls)
                os.makedirs(dest, exist_ok=True)
                for i, img_name in enumerate(split_imgs):
                    ext  = os.path.splitext(img_name)[1]
                    # Embed class, condition, split index in filename
                    new_name = f"{cls}_{cond}_{split_name}_{i:04d}{ext}"
                    shutil.copy(
                        os.path.join(src_path, img_name),
                        os.path.join(dest, new_name))

            print(f"  {cls:12s} {cond:6s}: "
                  f"{n_train} train / {n_val} val / {n_test} test  "
                  f"(capped from {len(os.listdir(src_path))} to {n})")

if not os.path.exists(SPLIT_DIR):
    print("Splitting originals (proportionate, capped)...\n")
    split_proportionate(ORIGINAL_DATA_DIR, SPLIT_DIR, seed=RANDOM_SEED)
    print("\nSplit complete.")
else:
    print("Split exists — skipping. "
          "Delete /content/split_originals/ to redo.")


# ─────────────────────────────────────────────────────────────
# CELL 5 — Augment Training Set (per condition)
# ─────────────────────────────────────────────────────────────
# %%
AUG_DIR = "/content/augmented_data"

aug_gen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

def augment_condition(cls, cond, split_dir, aug_dir,
                      target, seed=42):
    """
    Augment one class-condition combination to target count.
    Original images are included in target count.
    """
    src = os.path.join(split_dir, "train", cls)
    dst = os.path.join(aug_dir, "train", cls)
    os.makedirs(dst, exist_ok=True)

    # Get only images for this condition
    all_imgs = [f for f in os.listdir(src)
                if f.startswith(f"{cls}_{cond}_")]

    if not all_imgs:
        print(f"    WARNING: no {cond} images found for {cls}")
        return 0

    # Copy originals first
    for img_name in all_imgs:
        shutil.copy(os.path.join(src, img_name),
                    os.path.join(dst, img_name))

    n_orig   = len(all_imgs)
    n_needed = target - n_orig

    if n_needed <= 0:
        return n_orig

    # Generate augmented images
    np.random.seed(seed)
    count    = 0
    img_cycle = 0

    while count < n_needed:
        img_name = all_imgs[img_cycle % n_orig]
        img_cycle += 1

        img_path = os.path.join(src, img_name)
        img      = load_img(img_path, target_size=IMG_SIZE)
        x        = img_to_array(img).reshape((1,) +
                   img_to_array(img).shape)

        for batch in aug_gen.flow(x, batch_size=1,
                                  seed=seed + count):
            ext      = os.path.splitext(img_name)[1]
            aug_name = f"{cls}_{cond}_aug_{count:05d}{ext}"
            array_to_img(batch[0]).save(
                os.path.join(dst, aug_name))
            count += 1
            break

    return n_orig + count


if not os.path.exists(AUG_DIR):
    print(f"Augmenting each condition to "
          f"{AUG_TARGET_PER_CONDITION} images...\n")
    os.makedirs(AUG_DIR, exist_ok=True)

    for cls in CLASSES:
        print(f"\n{cls.upper()}")
        for cond in CONDITIONS:
            total = augment_condition(
                cls, cond,
                SPLIT_DIR, AUG_DIR,
                AUG_TARGET_PER_CONDITION,
                seed=RANDOM_SEED
            )
            print(f"  {cond:6s}: {total} images "
                  f"({AUG_TARGET_PER_CONDITION} target)")

    # Copy val and test as-is (no augmentation)
    print("\nCopying val and test (no augmentation)...")
    for split in ["val", "test"]:
        for cls in CLASSES:
            src = os.path.join(SPLIT_DIR, split, cls)
            dst = os.path.join(AUG_DIR, split, cls)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)

    print("\nAugmentation complete.")
else:
    print("Augmented data exists — skipping.")

# Print final counts
print("\nFinal dataset summary:")
grand_total = 0
for split in ["train", "val", "test"]:
    split_total = 0
    for cls in CLASSES:
        cls_dir = os.path.join(AUG_DIR, split, cls)
        if os.path.exists(cls_dir):
            n = len([f for f in os.listdir(cls_dir)
                     if not f.startswith('.')])
            split_total += n
    grand_total += split_total
    print(f"  {split:6s}: {split_total} images "
          f"({split_total // len(CLASSES)} per class)")
print(f"  TOTAL : {grand_total} images")


# ─────────────────────────────────────────────────────────────
# CELL 6 — Data Generators
# ─────────────────────────────────────────────────────────────
# %%
TRAIN_DIR = os.path.join(AUG_DIR, "train")
VAL_DIR   = os.path.join(AUG_DIR, "val")
TEST_DIR  = os.path.join(AUG_DIR, "test")

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=CLASSES,
    shuffle=True, seed=RANDOM_SEED)

val_gen = datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=CLASSES, shuffle=False)

test_gen = datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=CLASSES, shuffle=False)

print(f"Training  : {train_gen.samples} images")
print(f"Validation: {val_gen.samples} images")
print(f"Test      : {test_gen.samples} images")
print(f"Classes   : {train_gen.class_indices}")


# ─────────────────────────────────────────────────────────────
# CELL 7 — Build Model
# ─────────────────────────────────────────────────────────────
# %%
def build_model(num_classes=5):
    base = InceptionV3(weights="imagenet", include_top=False,
                       input_shape=(299, 299, 3))
    base.trainable = False

    x      = base.output
    x      = GlobalAveragePooling2D()(x)
    x      = Dense(128, activation="relu")(x)
    x      = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    return model

model = build_model()
model.summary()


# ─────────────────────────────────────────────────────────────
# CELL 8 — Train
# ─────────────────────────────────────────────────────────────
# %%
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.keras")

history = model.fit(
    train_gen, epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True,
                        monitor="val_accuracy", verbose=1)
    ],
    verbose=1
)
print(f"\nTraining complete. Model saved: {MODEL_PATH}")


# ─────────────────────────────────────────────────────────────
# CELL 9 — Training Curves
# ─────────────────────────────────────────────────────────────
# %%
def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')
    ep = range(1, len(history.history['accuracy']) + 1)

    ax1.plot(ep, history.history['accuracy'],
             'b-o', ms=4, lw=2, label='Train')
    ax1.plot(ep, history.history['val_accuracy'],
             'r-s', ms=4, lw=2, label='Validation')
    ax1.set_title('Accuracy History', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep, history.history['loss'],
             'b-o', ms=4, lw=2, label='Train')
    ax2.plot(ep, history.history['val_loss'],
             'r-s', ms=4, lw=2, label='Validation')
    ax2.set_title('Loss History', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.show()
    print(f"Saved: {save_path}")

plot_training_curves(
    history,
    os.path.join(RESULTS_DIR, "fig_training_curves.png"))


# ─────────────────────────────────────────────────────────────
# CELL 10 — Test Evaluation + Per-Class Table
# ─────────────────────────────────────────────────────────────
# %%
best_model = tf.keras.models.load_model(MODEL_PATH)

test_gen.reset()
test_loss, test_acc = best_model.evaluate(test_gen, verbose=1)
print(f"\nTest Accuracy : {test_acc*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

test_gen.reset()
y_pred_probs = best_model.predict(test_gen, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = test_gen.classes
filenames    = test_gen.filenames

report_text = classification_report(
    y_true, y_pred, target_names=CLASSES)
report_dict = classification_report(
    y_true, y_pred, target_names=CLASSES, output_dict=True)
print(f"\n{report_text}")

# Per-class table
cm   = confusion_matrix(y_true, y_pred)
rows = []
for i, cls in enumerate(CLASSES):
    rows.append({
        "Class"         : cls.capitalize(),
        "Correct/Total" : f"{cm[i,i]}/{cm[i].sum()}",
        "Precision"     : f"{report_dict[cls]['precision']:.3f}",
        "Recall"        : f"{report_dict[cls]['recall']:.3f}",
        "F1-Score"      : f"{report_dict[cls]['f1-score']:.3f}"
    })
rows.append({
    "Class"         : "Macro Average",
    "Correct/Total" : f"{int(np.diag(cm).sum())}/{cm.sum()}",
    "Precision"     : f"{report_dict['macro avg']['precision']:.3f}",
    "Recall"        : f"{report_dict['macro avg']['recall']:.3f}",
    "F1-Score"      : f"{report_dict['macro avg']['f1-score']:.3f}"
})
df_perclass = pd.DataFrame(rows)
print("\nPer-Class Table:\n")
print(df_perclass.to_string(index=False))

with open(os.path.join(RESULTS_DIR, "classification_report.txt"),
          "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write(report_text)
df_perclass.to_csv(
    os.path.join(RESULTS_DIR, "per_class_results.csv"), index=False)
print("\nSaved classification report and per-class CSV.")


# ─────────────────────────────────────────────────────────────
# CELL 11 — Condition-Wise Accuracy (from filenames)
# ─────────────────────────────────────────────────────────────
# %%
def get_condition(filename):
    f = os.path.basename(filename).lower()
    if "_light_" in f:
        return "Lightly Occluded"
    elif "_heavy_" in f or "_sand_" in f:
        return "Heavily Occluded"
    elif "_clean_" in f:
        return "Clean / Wet"
    return "Unknown"

results_df = pd.DataFrame({
    "filename"  : filenames,
    "true"      : y_true,
    "pred"      : y_pred,
    "correct"   : y_true == y_pred
})
results_df["condition"] = results_df["filename"].apply(get_condition)

print("\nCondition-wise Accuracy:\n")
condition_results = {}
for cond in ["Clean / Wet", "Lightly Occluded", "Heavily Occluded"]:
    subset = results_df[results_df["condition"] == cond]
    if len(subset) > 0:
        acc = subset["correct"].mean() * 100
        condition_results[cond] = acc
        print(f"  {cond:22s}: {acc:.2f}%  ({len(subset)} images)")
    else:
        print(f"  {cond:22s}: 0 images — check filenames")

pd.DataFrame([{"Condition": k, "Accuracy (%)": round(v, 2)}
              for k, v in condition_results.items()]).to_csv(
    os.path.join(RESULTS_DIR, "condition_accuracy.csv"), index=False)
print("\nCondition accuracy saved.")


# ─────────────────────────────────────────────────────────────
# CELL 12 — Condition Bar Chart
# ─────────────────────────────────────────────────────────────
# %%
def plot_condition_bar(condition_results, save_path):
    conditions = list(condition_results.keys())
    accuracies = list(condition_results.values())
    colors     = ['#2196F3', '#7B61FF', '#FF6F00']

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')
    bars = ax.bar(conditions, accuracies, color=colors,
                  width=0.5, edgecolor='white', linewidth=1.5)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.15,
                f'{acc:.2f}%', ha='center', va='bottom',
                fontsize=13, fontweight='bold',
                color=bar.get_facecolor())

    min_acc = min(accuracies)
    ax.set_ylim([max(0, min_acc - 5), 101])
    ax.set_ylabel('Classification Accuracy (%)', fontsize=13)
    ax.set_title('Classification Accuracy Across Imaging Conditions',
                 fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('#FAFAFA')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.show()
    print(f"Saved: {save_path}")

if condition_results:
    plot_condition_bar(
        condition_results,
        os.path.join(RESULTS_DIR, "fig_condition_accuracy.png"))


# ─────────────────────────────────────────────────────────────
# CELL 13 — Confusion Matrix
# ─────────────────────────────────────────────────────────────
# %%
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('white')
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[c.capitalize() for c in CLASSES],
            yticklabels=[c.capitalize() for c in CLASSES],
            linewidths=0.5, linecolor='white', ax=ax,
            annot_kws={"size": 13})
ax.set_xlabel('Predicted Label', fontsize=13, labelpad=10)
ax.set_ylabel('True Label', fontsize=13, labelpad=10)
ax.set_title('Confusion Matrix', fontsize=15,
             fontweight='bold', pad=15)
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "fig_confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved: {cm_path}")


# ─────────────────────────────────────────────────────────────
# CELL 14 — ROC Curves
# ─────────────────────────────────────────────────────────────
# %%
y_true_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
colors     = ['#2196F3','#FF6F00','#4CAF50','#E91E63','#9C27B0']

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('white')

auc_scores = {}
for i, (cls, col) in enumerate(zip(CLASSES, colors)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc     = auc(fpr, tpr)
    auc_scores[cls] = roc_auc
    ax.plot(fpr, tpr, color=col, linewidth=2,
            label=f'{cls.capitalize()} (AUC = {roc_auc:.3f})')

ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.5,
        label='Random classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — One-vs-Rest',
             fontsize=14, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(RESULTS_DIR, "fig_roc_curves.png")
plt.savefig(roc_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved: {roc_path}")
print("\nAUC Scores:")
for cls, score in auc_scores.items():
    print(f"  {cls.capitalize():12s}: {score:.3f}")


# ─────────────────────────────────────────────────────────────
# CELL 15 — K-Fold Cross Validation
# ─────────────────────────────────────────────────────────────
# %%
print(f"Running {NUM_FOLDS}-fold cross-validation...\n")

cv_paths, cv_labels = [], []
for label_idx, cls in enumerate(CLASSES):
    cls_dir = os.path.join(TRAIN_DIR, cls)
    if os.path.exists(cls_dir):
        imgs = [os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        cv_paths.extend(imgs)
        cv_labels.extend([label_idx] * len(imgs))

cv_paths  = np.array(cv_paths)
cv_labels = np.array(cv_labels)
print(f"Total training images for CV: {len(cv_paths)}")

kf           = KFold(n_splits=NUM_FOLDS, shuffle=True,
                     random_state=RANDOM_SEED)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(cv_paths)):
    print(f"\nFold {fold+1}/{NUM_FOLDS}")

    for cls in CLASSES:
        os.makedirs(f"/content/cv_{fold}/train/{cls}", exist_ok=True)
        os.makedirs(f"/content/cv_{fold}/val/{cls}",   exist_ok=True)

    for idx in train_idx:
        cls_name = CLASSES[cv_labels[idx]]
        shutil.copy(cv_paths[idx],
                    f"/content/cv_{fold}/train/{cls_name}/"
                    + os.path.basename(cv_paths[idx]))
    for idx in val_idx:
        cls_name = CLASSES[cv_labels[idx]]
        shutil.copy(cv_paths[idx],
                    f"/content/cv_{fold}/val/{cls_name}/"
                    + os.path.basename(cv_paths[idx]))

    ft = datagen.flow_from_directory(
        f"/content/cv_{fold}/train", target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical",
        classes=CLASSES, shuffle=True, seed=RANDOM_SEED)
    fv = datagen.flow_from_directory(
        f"/content/cv_{fold}/val", target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical",
        classes=CLASSES, shuffle=False)

    fm = build_model()
    fm.fit(ft, epochs=EPOCHS, validation_data=fv,
           callbacks=[EarlyStopping(monitor="val_loss", patience=5,
                                    restore_best_weights=True,
                                    verbose=0)],
           verbose=1)

    _, fold_acc = fm.evaluate(fv, verbose=0)
    fv.reset()
    fp = np.argmax(fm.predict(fv, verbose=0), axis=1)
    fr = classification_report(fv.classes, fp,
                                target_names=CLASSES,
                                output_dict=True, zero_division=0)
    fold_results.append({
        "fold"      : fold + 1,
        "accuracy_%": round(fold_acc * 100, 2),
        "precision" : round(fr["macro avg"]["precision"], 3),
        "recall"    : round(fr["macro avg"]["recall"], 3),
        "f1"        : round(fr["macro avg"]["f1-score"], 3)
    })
    print(f"  Fold {fold+1}: {fold_acc*100:.2f}%")
    shutil.rmtree(f"/content/cv_{fold}")

df_kfold = pd.DataFrame(fold_results)
avg = {
    "fold"      : "Average",
    "accuracy_%": round(df_kfold["accuracy_%"].mean(), 2),
    "precision" : round(df_kfold["precision"].mean(), 3),
    "recall"    : round(df_kfold["recall"].mean(), 3),
    "f1"        : round(df_kfold["f1"].mean(), 3)
}
df_kfold_display = pd.concat(
    [df_kfold, pd.DataFrame([avg])], ignore_index=True)

print(f"\nK-Fold Results:\n")
print(df_kfold_display.to_string(index=False))
df_kfold_display.to_csv(
    os.path.join(RESULTS_DIR, "kfold_results.csv"), index=False)


# ─────────────────────────────────────────────────────────────
# CELL 16 — K-Fold Chart
# ─────────────────────────────────────────────────────────────
# %%
folds  = [f"Fold {r['fold']}" for _, r in df_kfold.iterrows()]
accs   = list(df_kfold["accuracy_%"])
avg_ac = df_kfold["accuracy_%"].mean()

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('white')
bars = ax.bar(folds, accs,
              color=['#2196F3','#4CAF50','#FF6F00'],
              width=0.45, edgecolor='white', linewidth=1.5)
ax.axhline(y=avg_ac, color='red', linestyle='--',
           linewidth=1.5, label=f'Average: {avg_ac:.2f}%')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f'{acc:.2f}%', ha='center', va='bottom',
            fontsize=12, fontweight='bold')
ax.set_ylim([max(0, min(accs) - 5), 101])
ax.set_ylabel('Validation Accuracy (%)', fontsize=13)
ax.set_title(f'{NUM_FOLDS}-Fold Cross-Validation Accuracy',
             fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_facecolor('#FAFAFA')
plt.tight_layout()
kf_path = os.path.join(RESULTS_DIR, "fig_kfold_accuracy.png")
plt.savefig(kf_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved: {kf_path}")


# ─────────────────────────────────────────────────────────────
# CELL 17 — Save All Results to Excel
# ─────────────────────────────────────────────────────────────
# %%
excel_path = os.path.join(RESULTS_DIR, "all_results_summary.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    pd.DataFrame(history.history).to_excel(
        writer, sheet_name="Training History", index_label="Epoch")
    pd.DataFrame([{
        "Test Accuracy (%)": round(test_acc * 100, 2),
        "Test Loss": round(test_loss, 4)
    }]).to_excel(writer, sheet_name="Test Accuracy", index=False)
    df_perclass.to_excel(
        writer, sheet_name="Per-Class Performance", index=False)
    df_kfold_display.to_excel(
        writer, sheet_name="K-Fold Results", index=False)
    pd.DataFrame([{"Condition": k, "Accuracy (%)": round(v, 2)}
                  for k, v in condition_results.items()]).to_excel(
        writer, sheet_name="Condition Accuracy", index=False)
    pd.DataFrame([{"Class": k, "AUC": round(v, 3)}
                  for k, v in auc_scores.items()]).to_excel(
        writer, sheet_name="AUC Scores", index=False)
print(f"All results saved: {excel_path}")


# ─────────────────────────────────────────────────────────────
# CELL 18 — Final Summary
# ─────────────────────────────────────────────────────────────
# %%
print("=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\nOriginals per condition (capped): {MIN_PER_CONDITION}")
print(f"Augmented train per condition   : {AUG_TARGET_PER_CONDITION}")
print(f"Training images                 : {train_gen.samples}")
print(f"Validation images               : {val_gen.samples}")
print(f"Test images                     : {test_gen.samples}")
print(f"\nTest Accuracy    : {test_acc*100:.2f}%")
print(f"Macro Precision  : {report_dict['macro avg']['precision']:.3f}")
print(f"Macro Recall     : {report_dict['macro avg']['recall']:.3f}")
print(f"Macro F1         : {report_dict['macro avg']['f1-score']:.3f}")
print(f"\nK-Fold Average   : {df_kfold['accuracy_%'].mean():.2f}%")
print(f"\nCondition Accuracy:")
for cond, acc in condition_results.items():
    print(f"  {cond:22s}: {acc:.2f}%")
print(f"\nAUC Scores:")
for cls, score in auc_scores.items():
    print(f"  {cls.capitalize():12s}: {score:.3f}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# CELL 19 — Download Everything
# ─────────────────────────────────────────────────────────────
# %%
from google.colab import files

DOWNLOAD_ZIP = "/content/CG_Results_All.zip"
with zipfile.ZipFile(DOWNLOAD_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(RESULTS_DIR):
        zf.write(os.path.join(RESULTS_DIR, fname), arcname=fname)

print(f"Zipped: {os.path.getsize(DOWNLOAD_ZIP)/1024/1024:.1f} MB")
files.download(DOWNLOAD_ZIP)
