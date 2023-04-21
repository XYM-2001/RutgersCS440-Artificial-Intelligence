import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate random image data
train_images = np.random.rand(100, 28, 28, 1)
test_images = np.random.rand(20, 28, 28, 1)

# Generate random binary labels
train_labels = np.random.randint(2, size=100)
test_labels = np.random.randint(2, size=20)

# Split the data into training and testing sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

print("Shape of train_images:", train_images.shape)
print("Shape of train_labels:", train_labels.shape)
print("Shape of val_images:", val_images.shape)
print("Shape of val_labels:", val_labels.shape)
print("Shape of test_images:", test_images.shape)
print("Shape of test_labels:", test_labels.shape)

# Display the first 5 images from the training set
fig, axs = plt.subplots(1, 5, figsize=(15,15))
for i in range(5):
    axs[i].imshow(train_images[i].squeeze(), cmap='gray')
    axs[i].axis('off')
    axs[i].set_title('Label: {}'.format(train_labels[i]))
plt.show()

# Define the model
model = keras.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
