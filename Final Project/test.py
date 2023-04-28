import tensorflow as tf
import numpy as np
import os


# Load digit data
__location__ = os.path.dirname(__file__)
train_images = np.loadtxt(os.path.join(__location__, 'data\\digitdata\\trainingimages'), dtype=str)
train_labels = np.loadtxt(os.path.join(__location__, 'data\\digitdata\\traininglabels'))

test_images = np.loadtxt(os.path.join(__location__, 'data\\digitdata\\testimages'), dtype=str)
test_labels = np.loadtxt(os.path.join(__location__, 'data\\digitdata\\testlabels'))

# # Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=10)

# Test model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Load face data
train_images = np.loadtxt('facedatatrain.txt', dtype=str, delimiter='\n')
train_labels = np.loadtxt('facedatatrainlabels.txt')

test_images = np.loadtxt('facedatatest.txt', dtype=str, delimiter='\n')
test_labels = np.loadtxt('facedatatestlabels.txt')

# Reshape face data
train_images = train_images.reshape((len(train_images), 70, 60, 1))
test_images = test_images.reshape((len(test_images), 70, 60, 1))

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(70, 60, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=10)

# Test model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)