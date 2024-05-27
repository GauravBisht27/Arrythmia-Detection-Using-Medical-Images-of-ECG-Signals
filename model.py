import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf


train_folder_path = "C:\\Users\\Dipesh Singh\\Downloads\\train\\train"
test_folder_path ="C:\\Users\\Dipesh Singh\\Downloads\\test\\test"


# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # Use 4 units for 4 classes and softmax activation
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(train_folder_path, target_size=(128, 128),
                                                    batch_size=10, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(train_folder_path, target_size=(128, 128),
                                                        batch_size=10, class_mode='categorical')

model.fit(train_generator, steps_per_epoch=20, epochs=30, validation_data=validation_generator)
model.save("model.h5")
