import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import time
import numpy as np
import math


# Paths to dataset directories
base_dir = '/home/koczka/Documents/ANN/processed_images'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)  # Normalization for validation data
test_datagen = ImageDataGenerator(rescale=1./255)  # Normalization for test data

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=10,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=10,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=10,
    class_mode='binary',
    shuffle=False
)

# Convert the generators to tf.data.Datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 128, 128, 3], [None])
).repeat()

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 128, 128, 3], [None])
).repeat()

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 128, 128, 3], [None])
)

model = tf.keras.models.load_model('model_1.keras')




# Evaluate model on the test set using test_dataset
val_steps_per_epoch = math.ceil(val_generator.samples / val_generator.batch_size)
val_results = model.evaluate(val_dataset, steps=val_steps_per_epoch)
print(f"Validation results for model {1}: Loss = {val_results[0]}, Accuracy = {val_results[1]}")
# Predict on the test set
predictions = model.predict(val_dataset, steps=val_steps_per_epoch)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]

true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Compute confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Model 1')
plt.savefig('confusion_matrix_1_validation.png')
plt.close()