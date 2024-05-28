import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ścieżki do katalogów zestawów danych
base_dir = '/home/koczka/Documents/ANN/processed_images'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Wstępne przetwarzanie danych i augmentacja
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalizacja dla danych treningowych
val_datagen = ImageDataGenerator(rescale=1./255)    # Normalizacja dla danych walidacyjnych
test_datagen = ImageDataGenerator(rescale=1./255)   # Normalizacja dla danych testowych

# Generator danych
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=10,
    class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=10,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=10,
    class_mode='binary')


layer_1 = [3,6,10,15,32,45,60,75,90,100]
layer_2 = [6,12,20,30,64,90,120,150,180,200]
layer_3 = [12,24,40,60,128,180,240,300,360,400]
density = [12,24,40,60,128,180,240,300,360,400]

max_models = 10

# Loop for trying different models
for i in range(max_models):
    # Build model
    model = models.Sequential([
    Input(shape=(128, 128, 3)),
    layers.Conv2D(layer_1[i], (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(layer_2[i], (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(layer_3[i], (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(density[i], activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

    # Kompilacja modelu
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Record start time
    start_time = time.time()
    


    # more data for training, 
    # validation after 5 epochs
    # change architecture to the one that was used with this dataset 
    
    # try data augmentation



    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=5,  # Dostosuj liczby zależnie od liczby danych
        epochs=10,
        validation_data=val_generator,
        validation_steps=1)
    # Record end time
    end_time = time.time()
    
    # Calculate training time
    training_time = end_time - start_time

    # Ocena modelu na danych testowych
    test_loss, test_acc = model.evaluate(test_generator, steps=10)
    print(f'Test accuracy: {test_acc}')

    # Save model
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
    
    # Record model performance
    with open('model_performance.txt', 'a') as f: f.write(f"Model {i+1}: architecture: layer_1 = {layer_1[i]},layer_2 = {layer_2[i]},layer_3 = {layer_3[i]},density = {density[i]}, Batch size - 50, Optimizer - Adam, Training Time - {training_time} seconds\n")

    # # Trenowanie modelu
    # history = model.fit(
    #     train_generator,
    #     steps_per_epoch=15,  # Dostosuj liczby zależnie od liczby danych
    #     epochs=10,
    #     validation_data=val_generator,
    #     validation_steps=5)

    # # Ocena modelu na danych testowych
    # test_loss, test_acc = model.evaluate(test_generator, steps=10)
    # print(f'Test accuracy: {test_acc}')
