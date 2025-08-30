```py

# ============================
# STEP 1: Setup
# ============================
import numpy as np
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# If dataset is in Google Drive, mount it
from google.colab import drive
drive.mount('/content/drive')

# Example: Dataset in Google Drive
# Change path to where your dataset is stored
source_dir = "/content/drive/MyDrive/datasets/alzheimers/combined_images"

# Output folders for clean splits (inside Colab runtime)
train_dir = "/content/train_images"
test_dir = "/content/test_images"

# Create output folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split into train/test
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    files = os.listdir(class_path)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    for fname in train_files:
        shutil.copyfile(os.path.join(class_path, fname), os.path.join(train_class_dir, fname))
    for fname in test_files:
        shutil.copyfile(os.path.join(class_path, fname), os.path.join(test_class_dir, fname))

# ============================
# STEP 2: Data Generators
# ============================
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.08)

train_generator = train_val_datagen.flow_from_directory(
    train_dir,
    target_size=(180,180),
    batch_size=64,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_val_datagen.flow_from_directory(
    train_dir,
    target_size=(180,180),
    batch_size=128,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(180,180),
    batch_size=128,
    class_mode='categorical',
    shuffle=False
)

# ============================
# STEP 3: Model
# ============================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", activation="relu", 
                 kernel_regularizer=regularizers.l2(0.001), input_shape=(180,180,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3,3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3,3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(4, activation="softmax"))

model.compile(optimizer=Adam(learning_rate=0.0007),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=["accuracy"])

model.summary()

# ============================
# STEP 4: Training
# ============================
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(filepath='/content/best_model.h5', monitor='val_accuracy', 
                    save_best_only=True, mode='max', verbose=1)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# ============================
# STEP 5: Plot Curves
# ============================
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend(); plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend(); plt.show()

# ============================
# STEP 6: Evaluation + Confusion Matrix + ROC
# ============================
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

loss_test, acc_test = model.evaluate(test_generator)
print(f"Test Loss = {loss_test:.4f}, Test Accuracy = {acc_test:.4f}")

y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
plt.show()

# ROC Curve
y_true_bin = label_binarize(y_true, classes=range(len(class_labels)))
plt.figure(figsize=(10,8))
for i in range(len(class_labels)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_labels[i]} (AUC = {roc_auc:.2f})')

plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("Multi-Class ROC-AUC")
plt.legend(); plt.show()

```


# Input feild

```py

from google.colab import files
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Upload an image
uploaded = files.upload()

for fname in uploaded.keys():
    img_path = fname
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Prediction
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    pred_label = class_labels[pred_class]
    confidence = preds[0][pred_class] * 100

    # Show result
    plt.imshow(image.load_img(img_path))
    plt.axis("off")
    plt.title(f"Prediction: {pred_label} ({confidence:.2f}%)")
    plt.show()


```