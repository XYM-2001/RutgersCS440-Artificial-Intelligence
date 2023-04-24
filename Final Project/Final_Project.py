import os
import numpy as np

def display_image(image):
    for i in image:
        print(i)

class Perceptron:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features)

    def predict(self, features):
        weighted_sum = np.dot(features, self.weights)

        if weighted_sum >= 0:
            return 1
        else:
            return 0
    
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
        
    print(perceptron.weights)
    return perceptron

def main():
    labels = load_imagelabel('traininglabels')
    labels = [int(i) for i in labels]
    images = load_image('trainingimages', len(labels))
    training_set = images[0:500]
    training_labels = labels[0:500]
    testing_set = images[500:600]
    testing_labels = labels[500:600]
    perceptron = Perceptron(num_features=28*28)
    print(perceptron.weights)
    perceptron = train_model(training_set, training_labels, perceptron)
    false = 0
    for i in range(len(testing_set)):
        features = convert_Integer(testing_set[i]).flatten()/255.0
        prediction = perceptron.predict(features)
        if prediction != testing_labels[i]:
            false += 1
    print(false/len(testing_set))

if __name__=="__main__":
    main()
