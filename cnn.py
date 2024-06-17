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
    rescale=1./255,  # Normalization for training data
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

# Parameters list
layer_1 = [10, 20, 30]
layer_2 = [20, 40, 60]
layer_3 = [40, 80, 120]
layer_4 = [40, 80, 120]
layer_5 = [40, 80, 120]
density = [40, 80, 120]

max_models = 3

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Loop for trying different models
for i in range(max_models):
    # Build model
    model = Sequential([
        Input(shape=(128, 128, 3)),
        Conv2D(layer_1[i], kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(layer_2[i], (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(layer_3[i], (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(layer_4[i], (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(layer_5[i], (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(density[i], activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Record start time
    start_time = time.time()
    
    # Train model with early stopping
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=100,
        validation_data=val_dataset,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=[early_stopping]
    )
    
    # Record end time
    end_time = time.time()
    
    # Calculate training time
    training_time = end_time - start_time
        
    model.save(f'model_{i+1}.keras')
        
    # Plot learning and test loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning and Test Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'learning_test_loss_curves_{i+1}.png')
    plt.close()

    # Evaluate model on the test set using test_dataset
    test_steps_per_epoch = math.ceil(test_generator.samples / test_generator.batch_size)
    test_results = model.evaluate(test_dataset, steps=test_steps_per_epoch)
    print(f"Test results for model {i+1}: Loss = {test_results[0]}, Accuracy = {test_results[1]}")

    # Predict on the test set
    predictions = model.predict(test_dataset, steps=test_steps_per_epoch)
    predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]
    
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    # Compute confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Model {i+1}')
    plt.savefig(f'confusion_matrix_{i+1}.png')
    plt.close()
    
    # Compute classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0, output_dict=True)
    
    # Save confusion matrix and classification report to text file
    with open('model_performance.txt', 'a') as f:
        f.write(f"Model {i+1}:\n")
        f.write(f"Architecture: layer_1 = {layer_1[i]}, layer_2 = {layer_2[i]}, layer_3 = {layer_3[i]}, layer_4 = {layer_4[i]}, layer_5 = {layer_5[i]}, density = {density[i]}, Batch size = 10, Optimizer = Adam, Training Time = {training_time} seconds\n")
        f.write(f"Test results: Loss = {test_results[0]}, Accuracy = {test_results[1]}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Classification Report:\n")
        for label, metrics in report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                f.write(f"{label}: precision = {metrics['precision']:.4f}, recall = {metrics['recall']:.4f}, f1-score = {metrics['f1-score']:.4f}, support = {metrics['support']}\n")
        f.write("\n")
