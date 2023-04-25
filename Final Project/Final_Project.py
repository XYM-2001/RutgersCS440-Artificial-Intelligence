import os
import numpy as np

def display_image(image):
    for i in image:
        print(i)

class Perceptron:
    def __init__(self, num_features, digit):
        self.weights = np.zeros(num_features)
        self.w0 = 0
        self.digit = digit

    def predict(self, features):
        weighted_sum = self.w0 + np.dot(features, self.weights)

        return weighted_sum
    
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

def train_model(training_set, training_label, perceptrons):
    for i in range(len(training_set)):
        for i in range(10):
            perceptron = perceptrons[i]
            features = convert_Integer(training_set[i]).flatten()
            prediction = perceptron.predict(features)
            if prediction < 0 and training_label[i] == perceptron.digit:
                perceptron.weights = perceptron.weights + features
                perceptron.w0 += 1
            elif prediction >= 0 and  training_label[i] != perceptron.digit:
                perceptron.weights = perceptron.weights - features
                perceptron.w0 -= 1
            perceptrons[i] = perceptron
    return perceptrons

def main():
    traininglabels = load_imagelabel('traininglabels')
    traininglabels = [int(i) for i in traininglabels]
    trainingimages = load_image('trainingimages', len(traininglabels))
    testlabels = load_imagelabel('testlabels')
    testlabels = [int(i) for i in testlabels]
    testimages = load_image('testimages', len(testlabels))
    perceptrons = []
    for i in range(10):
        perceptrons.append(Perceptron(28*28, i))
    perceptrons = train_model(trainingimages, traininglabels, perceptrons)
    trues = 0
    nums = 0
    for i in range(len(testimages)):
        if testlabels[i] == 0:
            nums += 1
            features = convert_Integer(testimages[i]).flatten()
            prediction = perceptrons[0].predict(features)
            if prediction >= 0:
                trues += 1
    print(trues)
    print(nums)


if __name__=="__main__":
    main()
