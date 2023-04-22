import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
images = digits.images
targets = digits.target

# plot the first 5 images
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f"Label: {targets[i]}")
    ax.axis('off')
    
plt.show()