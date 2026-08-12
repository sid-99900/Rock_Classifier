import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. SETUP PATHS (Windows Style)
# Update this to the folder where your augmented data is stored
dataset_dir = r"C:\Users\Siddhant\Desktop\BTP_Final_Dataset"
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# Detect Classes
materials = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print(f"Detected Classes: {materials}")

# 2. DATA GENERATORS
# target_size is fixed at 299x299 for InceptionV3
target_size = (299, 299)
# Reduced batch_size for local RAM stability (Shared 16GB)
batch_size = 16 

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=target_size, batch_size=batch_size,
    class_mode='categorical', shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=target_size, batch_size=batch_size,
    class_mode='categorical', shuffle=False
)

# 3. BUILD MODEL
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x) 
predictions = Dense(len(materials), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers initially
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. CALLBACKS
checkpoint = ModelCheckpoint('local_best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 5. TRAINING
print("Starting local training... This will download InceptionV3 weights on the first run.")
history = model.fit(
    train_generator, 
    epochs=20, 
    validation_data=test_generator,
    callbacks=[checkpoint, early_stop]
)

# 6. EVALUATION & VISUALIZATION
# Training Curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title('Local Accuracy Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.title('Local Loss Curves')
plt.legend()
plt.show()

# Classification Metrics
test_generator.reset()
preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes
filenames = test_generator.filenames

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=materials, yticklabels=materials)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Local Run)')
plt.show()

# ROC Curves
plt.figure(figsize=(10, 8))
for i, mat in enumerate(materials):
    fpr, tpr, _ = roc_curve(y_true == i, preds[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{mat} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves (Local Run)')
plt.legend()
plt.show()

# 7. CONDITION-WISE ACCURACY (For Error Analysis)
results_df = pd.DataFrame({
    'filename': filenames,
    'true': y_true,
    'pred': y_pred
})

results_df['condition'] = results_df['filename'].apply(
    lambda x: 'clean' if 'clean' in x else ('light' if 'light' in x else 'heavy')
)

print("\n--- Local Condition-Wise Accuracy ---")
for cond in ['clean', 'light', 'heavy']:
    cond_data = results_df[results_df['condition'] == cond]
    if len(cond_data) > 0:
        acc = (cond_data['true'] == cond_data['pred']).mean() * 100
        print(f"{cond.upper()} condition accuracy: {acc:.2f}%")

print("\n✅ Training complete. Best local model saved as 'local_best_model.keras'")
