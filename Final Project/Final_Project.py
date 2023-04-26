import os
import numpy as np



def display_image(image):
    for i in image:
        print(i)

class Perceptron:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features)
        self.w0 = 0

    def predict(self, features):
        weighted_sum = self.w0 + np.dot(features, self.weights)

        return weighted_sum

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.var = np.zeros((len(self.classes), X.shape[1]))
        self.prior = np.zeros(len(self.classes))

        # Calculate the mean, variance, and prior probability for each class
        for i, c in enumerate(self.classes):
            X_c = X[c == y]
            self.mean[i, :] = np.mean(X_c, axis=0)
            self.var[i, :] = np.var(X_c, axis=0)
            self.prior[i] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        # Calculate the likelihood of each feature given each class
        likelihood = np.zeros((len(self.classes), X.shape[1]))
        for i, c in enumerate(self.classes):
            likelihood[i, :] = -0.5 * np.log(2 * np.pi * self.var[i, :])
            likelihood[i, :] -= 0.5 * ((X - self.mean[i, :]) ** 2) / self.var[i, :]

        # Calculate the posterior probability of each class
        posterior = np.zeros(len(self.classes))
        for i, c in enumerate(self.classes):
            posterior[i] = np.sum(likelihood[i, :]) + np.log(self.prior[i])

        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posterior)]
    
def load_label(filename):
    __location__ = os.path.dirname(__file__)
    f = open(os.path.join(__location__, 'data\\' + filename))
    temp = f.readlines()
    f.close()
    return temp

def load_image(filename, number, x, y):
    __location__ = os.path.dirname(__file__)
    f = open(os.path.join(__location__, 'data\\' + filename))
    temp = f.readlines()
    temp.reverse()
    items = []
    for i in range(number):
        data = []
        for j in range(y):
            data.append(list(temp.pop())[0:x])
            if not temp:
                print('Reached the end of file.')
                items.append(data)
                return items
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

def train_digit_model(training_set, training_label, perceptrons):
    for i in range(len(training_set)):
        predictions = []
        for j in range(10):
            features = convert_Integer(training_set[i]).flatten()
            predictions.append(perceptrons[j].predict(features))
        prediction = predictions.index(max(predictions))
        if prediction != training_label[i]:
            perceptrons[prediction].weights = perceptrons[prediction].weights - features
            perceptrons[prediction].w0 -= 1
            perceptrons[training_label[i]].weights = perceptrons[training_label[i]].weights + features
            perceptrons[training_label[i]].w0 += 1
    return perceptrons

def train_face_model(training_set, training_label, perceptron):
    for i in range(len(training_set)):
        features = convert_Integer(training_set[i]).flatten()
        prediction = perceptron.predict(features)
        if prediction < 0 and training_label[i] == 1:
            perceptron.weights = perceptron.weights + features
            perceptron.w0 += 1
        elif prediction >= 0 and training_label[i] == 0: 
            perceptron.weights = perceptron.weights - features
            perceptron.w0 -= 1
    return perceptron

def main():
# Perceptron for digit data
    # traininglabels = load_label('digitdata\\traininglabels')
    # traininglabels = [int(i) for i in traininglabels]
    # trainingimages = load_image('digitdata\\trainingimages', len(traininglabels), 28, 28)
    # testlabels = load_label('digitdata\\testlabels')
    # testlabels = [int(i) for i in testlabels]
    # testimages = load_image('digitdata\\testimages', len(testlabels), 28, 28)
    # perceptrons = []
    # for i in range(10):
    #     perceptrons.append(Perceptron(28*28))
    # perceptrons = train_digit_model(trainingimages, traininglabels, perceptrons)
    # trues = 0
    # for i in range(len(testimages)):
    #     features = convert_Integer(testimages[i]).flatten()
    #     predictions = []
    #     for j in range(10):
    #         predictions.append(perceptrons[j].predict(features))
    #     prediction = predictions.index(max(predictions))
    #     if prediction == testlabels[i]:
    #         trues += 1
    # print('precision for Perceptron on testing digits: ', trues/len(testlabels))

#Perceptron for face data
    # traininglabels = load_label('facedata\\facedatatrainlabels')
    # traininglabels = [int(i) for i in traininglabels]
    # trainingset = load_image('facedata\\facedatatrain', len(traininglabels), 60, 70)
    # testlabels = load_label('facedata\\facedatatestlabels')
    # testlabels = [int(i) for i in testlabels]
    # testset = load_image('facedata\\facedatatest', len(testlabels), 60, 70)
    # perceptron = Perceptron(60*70)
    # perceptron = train_face_model(trainingset, traininglabels, perceptron)
    # trues = 0
    # for i in range(len(testset)):
    #     features = convert_Integer(testset[i]).flatten()
    #     prediction = perceptron.predict(features)
    #     if (prediction >= 0 and testlabels[i] == 1) or (prediction < 0 and testlabels[i] == 0):
    #         trues += 1
    # print('precision for Perceptron on testing faces: ', trues/len(testlabels))
#Naive Bayes for digit data
    traininglabels = load_label('digitdata\\traininglabels')
    traininglabels = [int(i) for i in traininglabels]
    trainingimages = load_image('digitdata\\trainingimages', len(traininglabels), 28, 28)
    testlabels = load_label('digitdata\\testlabels')
    testlabels = [int(i) for i in testlabels]
    testimages = load_image('digitdata\\testimages', len(testlabels), 28, 28)
    classifier = NaiveBayes()
    classifier.fit(trainingimages, traininglabels)
    predictions = np.zeros(len(testimages))
    for i in range(len(predictions)):
        predictions[i] = classifier.predict(testimages[i, :])
    accuracy = np.mean(predictions == testlabels)
    print(accuracy)

if __name__=="__main__":
    main()
