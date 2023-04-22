import numpy as np

# Define the image as a list of strings
image = [
    "        ",
    "        ",
    "  ++#++ ",
    " +##### ",
    "+######+",
    "+#####+##",
    "+######++#+",
    "+###++##++#+",
    "++###+     ",
    "+##+ +     ",
    "+##+       ",
    "+##+       ",
    "+##+       ",
    "++#+       ",
    "+##+       ",
    "+##+       ",
    "+##+       ",
    "+##+++++###++",
    "+#########+",
    "+#######+",
    " ++###++",
    "        ",
    "        ",
]

# Convert the list of strings to a 2D numpy array of integers
digits = np.array([[1 if c == '+' or c == '#' else 0 for c in row] for row in image])

# Define the perceptron class
class Perceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        # Add a bias term to the input data
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Initialize weights to zeros
        self.weights = np.zeros(X.shape[1])

        # Train the perceptron
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                if y[i] * np.dot(self.weights, X[i]) <= 0:
                    self.weights += self.lr * y[i] * X[i]

    def predict(self, X):
        # Add a bias term to the input data
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Compute the dot product of the input data and the weights
        dot_product = np.dot(X, self.weights)

        # Compute the predicted class labels as 0 or 1
        y_pred = np.where(dot_product > 0, 1, 0)

        return y_pred

# Define the training data and labels
X_train = digits.reshape(1, -1)
print(digits)
print(X_train)
# y_train = np.array([0])

# # Create a perceptron object and fit the training data
# perceptron = Perceptron()
# perceptron.fit(X_train, y_train)

# # Predict the label of the digit image
# y_pred = perceptron.predict(X_train)

# # Print the predicted label
# print("Predicted label:", y_pred[0])