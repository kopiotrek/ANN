import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    class_mode='binary'
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

# # Define the CNN model
# model = models.Sequential([
#     Input(shape=(128, 128, 3)),
    
#     # First Convolutional Block
#     Conv2D(6, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     MaxPooling2D((2, 2)),
    
#     # Second Convolutional Block
#     Conv2D(12, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     MaxPooling2D((2, 2)),
    
#     # Third Convolutional Block
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     layers.BatchNormalization(),
    
#     # Fully Connected Layers
#     Flatten(),
#     Dropout(0.5),
#     Dense(32, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# Define the CNN model
model = models.Sequential([
    Input(shape=(128, 128, 3)),
    
    # First Convolutional Block
    Conv2D(6, (7, 7), activation='relu'),
    MaxPooling2D((3, 3)),
    Dense(6, activation='relu'),
    Dense(12, activation='relu'),
    Dense(32, activation='relu'),
    layers.BatchNormalization(),
    Flatten(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=val_dataset,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_dataset, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_acc}')

# Plot learning and test loss curves
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Learning and Test Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('learning_test_loss_curves_2.png')
plt.close()
