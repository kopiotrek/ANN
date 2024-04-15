import os
import tensorflow as tf
from tensorflow.keras import layers, models
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
    batch_size=20,
    class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=20,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=20,
    class_mode='binary')

# Zdefiniowanie modelu CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=15,  # Dostosuj liczby zależnie od liczby danych
    epochs=10,
    validation_data=val_generator,
    validation_steps=5)

# Ocena modelu na danych testowych
test_loss, test_acc = model.evaluate(test_generator, steps=10)
print(f'Test accuracy: {test_acc}')
