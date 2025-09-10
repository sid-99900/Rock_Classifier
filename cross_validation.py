import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

data_dir = "/path/to/the/dataset"
image_size = (224, 224)
batch_size = 32
epochs = 10
class_indices = {'augmented_Basalt': 0, 'augmented_Magnetite': 1}

filepaths = []
labels = []

for class_name in os.listdir(data_dir):
    class_folder = os.path.join(data_dir, class_name)
    if os.path.isdir(class_folder):
        for fname in os.listdir(class_folder):
            if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                filepaths.append(os.path.join(class_folder, fname))
                labels.append(class_indices[class_name])

filepaths = np.array(filepaths)
labels = np.array(labels)

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(filepaths, labels)):
    print(f"\n Fold {fold + 1} ")

    df_train = pd.DataFrame({'filename': filepaths[train_idx], 'class': labels[train_idx]})
    df_val = pd.DataFrame({'filename': filepaths[val_idx], 'class': labels[val_idx]})

    inv_class_indices = {v: k for k, v in class_indices.items()}
    df_train['class'] = df_train['class'].map(inv_class_indices)
    df_val['class'] = df_val['class'].map(inv_class_indices)

    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_dataframe(
        df_train, x_col='filename', y_col='class',
        target_size=image_size, class_mode='binary',
        batch_size=batch_size, shuffle=True, seed=42
    )
    val_gen = datagen.flow_from_dataframe(
        df_val, x_col='filename', y_col='class',
        target_size=image_size, class_mode='binary',
        batch_size=batch_size, shuffle=False
    )

    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    checkpoint_path = f"/best_model_fold_{fold + 1}.keras"
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks, verbose=2)

    val_gen.reset()
    preds = (model.predict(val_gen) > 0.5).astype(int).reshape(-1)
    true = val_gen.classes
    report = classification_report(true, preds, output_dict=True)
    print(classification_report(true, preds))
    fold_results.append(report)


