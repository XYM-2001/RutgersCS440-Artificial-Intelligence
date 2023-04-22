import os
import numpy as np

def display_image(image):
    for i in image:
        print(i)

class Perceptron:
    def __init__(self, num_features):
        # Initialize the weights and bias to small random values
        self.weights = np.random.randn(num_features)
        self.bias = np.random.randn()

    def predict(self, features):
        # Calculate the weighted sum of the features
        weighted_sum = np.dot(features, self.weights) + self.bias

        # Apply the step function to the weighted sum
        if weighted_sum >= 0:
            return 1
        else:
            return 0
        
# class Perceptron:
#     def __init__(self, lr=0.1, epochs=100):
#         self.lr = lr
#         self.epochs = epochs

#     def fit(self, X, y):
#         # Add a bias term to the input data
#         X = np.hstack((X, np.ones((X.shape[0], 1))))

#         # Initialize weights to zeros
#         self.weights = np.zeros(X.shape[1])

#         # Train the perceptron
#         for epoch in range(self.epochs):
#             for i in range(X.shape[0]):
#                 if y[i] * np.dot(self.weights, X[i]) <= 0:
#                     self.weights += self.lr * y[i] * X[i]

#     def predict(self, X):
#         # Add a bias term to the input data
#         X = np.hstack((X, np.ones((X.shape[0], 1))))

#         # Compute the dot product of the input data and the weights
#         dot_product = np.dot(X, self.weights)

#         # Compute the predicted class labels as 0 or 1
#         y_pred = np.where(dot_product > 0, 1, 0)

#         return y_pred
    
def load_imagelabel(filename):
    __location__ = os.path.dirname(__file__)
    f = open(os.path.join(__location__, 'data\\digitdata\\' + filename))
    temp = f.readlines()
    f.close()
    return temp

def load_image(filename, number):
    __location__ = os.path.dirname(__file__)
    f = open(os.path.join(__location__, 'data\\digitdata\\' + filename))
    temp = f.readlines()
    temp.reverse()
    items = []
    for i in range(number):
        data = []
        for j in range(28):
            data.append(list(temp.pop())[0:28])
            if not temp:
                print('Reached the end of file.')
                items.append(data)
                return items
        if len(data[0]) < 27:
            print('Truncating at %d examples (maximum)' % i)
            break
        items.append(data)
    f.close()
    return items

def convert_Integer(image):
    converted = []
    for row in image:
        converted_row = []
        for pixel in row:
            if pixel == '+':
                converted_row.append(1)
            elif pixel == '#':
                converted_row.append(2)
            else:
                converted_row.append(0)
        converted.append(converted_row)
    return np.array(converted)

def train_model(training_set, training_label, perceptron):
    for i in range(len(training_set)):
        features = convert_Integer(training_set[i]).flatten()/255.0
        prediction = perceptron.predict(features)
        error = np.abs(training_label[i] - prediction)
        perceptron.weights += error*training_set[i]
        perceptron.bias += error
    return perceptron

def main():
    labels = load_imagelabel('traininglabels')
    images = load_image('trainingimages', len(labels))
    perceptron = Perceptron(num_features=28*28)

    converted = convert_Integer(images[1])
    display_image(converted)
    features = converted.flatten() / 255.0
    perceptron = Perceptron(num_features=len(features))
    

if __name__=="__main__":
    main()