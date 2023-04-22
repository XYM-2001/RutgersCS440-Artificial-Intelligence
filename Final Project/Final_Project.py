import os
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Update weights and bias using the perceptron learning rule
        for _ in range(self.n_iterations):
            for i in range(n_samples):
                linear_output = np.dot(self.weights, X[i]) + self.bias
                y_pred = np.where(linear_output >= 0, 1, -1)
                
                # Update weights and bias if prediction is incorrect
                if y_pred != y[i]:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = np.where(linear_output >= 0, 1, -1)
        return y_pred

def display_image(image):
    for i in image:
        print(i)
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
            data.append(list(temp.pop())[0:27])
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

# def convert_Integer(image):
#     converted = []
#     for row in image:
#         for col in row:


def main():
    labels = load_imagelabel('traininglabels')
    images = load_image('trainingimages', len(labels))

if __name__=="__main__":
    main()