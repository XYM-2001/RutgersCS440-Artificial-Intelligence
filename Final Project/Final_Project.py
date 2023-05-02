import os
import numpy as np
import tensorflow as tf
import random
import time


def display_image(image):
    for i in image:
        print(i)

def select_sample(images, labels, sample_size):
    combined = list(zip(images, labels))
    sample = random.sample(combined, sample_size)
    sampleimage, samplelabel = zip(*sample)
    return list(sampleimage), list(samplelabel)

def image_split(training_set, training_labels, num_class):
    image_classes = [[] for _ in range(num_class)]
    for i in range(len(training_labels)):
        image_classes[training_labels[i]].append(convert_Integer(training_set[i]).flatten())
    return image_classes


class Perceptron:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features)
        self.w0 = 0

    def predict(self, features):
        weighted_sum = self.w0 + np.dot(features, self.weights)

        return weighted_sum

class NaiveBayes:
    def __init__(self, num_features, prior):
        self.p = []
        for i in range(3):
            self.p.append(np.zeros(num_features))
        self.prior = prior

    def fit(self, X):
        for image in X:
            for i in range(len(image)):
                self.p[image[i]][i] += 1
        self.p = [i/len(X) for i in self.p]


    def predict(self, X):
        prob = [0.00000001]*len(X)
        for i in range(len(X)):
            prob[i] = self.p[X[i]][i] + 0.001
        prob = [np.log(i) for i in prob]
        return np.sum(prob) + np.log(self.prior)
        

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
#loading datasets
    trainingdigitlabels = load_label('digitdata\\traininglabels')
    trainingdigitlabels = [int(i) for i in trainingdigitlabels]
    trainingdigitimages = load_image('digitdata\\trainingimages', len(trainingdigitlabels), 28, 28)
    testdigitlabels = load_label('digitdata\\testlabels')
    testdigitlabels = [int(i) for i in testdigitlabels]
    testdigitimages = load_image('digitdata\\testimages', len(testdigitlabels), 28, 28)
    trainingfacelabels = load_label('facedata\\facedatatrainlabels')
    trainingfacelabels = [int(i) for i in trainingfacelabels]
    trainingfaceimages = load_image('facedata\\facedatatrain', len(trainingfacelabels), 60, 70)
    testfacelabels = load_label('facedata\\facedatatestlabels')
    testfacelabels = [int(i) for i in testfacelabels]
    testfaceimages = load_image('facedata\\facedatatest', len(testfacelabels), 60, 70)

# Perceptron for digit data
#     for samplesize in [20,40,60,80,100]:
#         accuracy = []
#         speed = []
#         for _ in range(5):
#             perceptrons = []
#             for i in range(10):
#                 perceptrons.append(Perceptron(28*28))
#             sampleimages, samplelabels = select_sample(trainingdigitimages, trainingdigitlabels, int(len(trainingdigitlabels)*samplesize/100))
#             starttime = time.time()
#             perceptrons = train_digit_model(sampleimages, samplelabels, perceptrons)
#             endtime = time.time()-starttime
#             speed.append(endtime)
#             trues = 0
#             for i in range(len(testdigitimages)):
#                 features = convert_Integer(testdigitimages[i]).flatten()
#                 predictions = []
#                 for j in range(10):
#                     predictions.append(perceptrons[j].predict(features))
#                 prediction = predictions.index(max(predictions))
#                 if prediction == testdigitlabels[i]:
#                     trues += 1
#             accuracy.append(trues/len(testdigitlabels))
#         print('accuracy for Perceptron on ', samplesize, 'percent digit image for 5 iterations: mean-', 
#               sum(accuracy)/len(accuracy), ' standard deviation-', np.std(accuracy))
#         print('time spent for training Perceptron on ', samplesize, 'percent digit image for 5 iterations: mean-', 
#               sum(speed)/len(speed), 'standard deviation-', np.std(speed))

# #Perceptron for face data
#     for samplesize in [20,40,60,80,100]:
#         accuracy = []
#         speed = []
#         for _ in range(5):
#             perceptron = Perceptron(60*70)
#             sampleimages, samplelabels = select_sample(trainingfaceimages, trainingfacelabels, int(len(trainingfacelabels)*samplesize/100))
#             starttime = time.time()
#             perceptron = train_face_model(sampleimages, samplelabels, perceptron)
#             endtime = starttime-time.time()
#             trues = 0
#             for i in range(len(testfaceimages)):
#                 features = convert_Integer(testfaceimages[i]).flatten()
#                 prediction = perceptron.predict(features)
#                 if (prediction >= 0 and testfacelabels[i] == 1) or (prediction < 0 and testfacelabels[i] == 0):
#                     trues += 1
#             accuracy.append(trues/len(testfacelabels))
#             speed.append(endtime)
#         print('accuracy for Perceptron on ', samplesize, 'percent face image for 5 iterations: mean-', 
#               sum(accuracy)/len(accuracy), ' standard deviation-', np.std(accuracy))
#         print('time spent for training Perceptron on ', samplesize, 'percent face image for 5 iterations: mean-', 
#               sum(speed)/len(speed), 'standard deviation-', np.std(speed))


#Naive Bayes for digit data
    for samplesize in [20,40,60,80,100]:
        accuracy = []
        speed = []
        for _ in range(5):
            sampleimages, samplelabels = select_sample(trainingdigitimages, trainingdigitlabels, int(len(trainingdigitlabels)*samplesize/100))
            trainingclasses = image_split(sampleimages, samplelabels,10)
            classifier = []
            starttime = time.time()
            for i in range(10):
                classifier.append(NaiveBayes(28*28,len(trainingclasses[i])/len(sampleimages)))
            for i in range(10):
                classifier[i].fit(trainingclasses[i])
            endtime = starttime-time.time()
            trues = 0
            for i in range(len(testdigitimages)):
                features = convert_Integer(testdigitimages[i]).flatten()
                predictions = []
                for j in range(10):
                    predictions.append(classifier[j].predict(features))
                prediction = predictions.index(max(predictions))
                if prediction == testdigitlabels[i]:
                    trues += 1
            accuracy.append(trues/len(testdigitlabels))
            speed.append(endtime)
        print('accuracy for Naive Bayes Classifier on ', samplesize, 'percent digit image for 5 iterations: mean-', 
              sum(accuracy)/len(accuracy), ' standard deviation-', np.std(accuracy))
        print('time spent for fitting Naive Bayes Classifier on ', samplesize, 'percent digit image for 5 iterations: mean-', 
              sum(speed)/len(speed), 'standard deviation-', np.std(speed))

#Naive Bayes for face data
    for samplesize in [20,40,60,80,100]:
        accuracy = []
        speed = []
        for _ in range(5):
            sampleimages, samplelabels = select_sample(trainingfaceimages, trainingfacelabels, int(len(trainingfacelabels)*samplesize/100))
            trainingclasses = image_split(sampleimages,samplelabels,2)
            classifier = []
            starttime = time.time()
            for i in range(2):
                classifier.append(NaiveBayes(60*70,len(trainingclasses[i])/len(sampleimages)))
            for i in range(2):
                classifier[i].fit(trainingclasses[i])
            endtime = starttime-time.time()
            trues = 0
            for i in range(len(testfaceimages)):
                predictions = []
                predictions.append(classifier[0].predict(convert_Integer(testfaceimages[i]).flatten()))
                predictions.append(classifier[1].predict(convert_Integer(testfaceimages[i]).flatten()))
                prediction = predictions.index(max(predictions))
                if prediction == testfacelabels[i]:
                    trues += 1
            accuracy.append(trues/len(testfacelabels))
            speed.append(endtime)
        print('accuracy for Naive Bayes Classifier on ', samplesize, 'percent face image for 5 iterations: mean-', 
              sum(accuracy)/len(accuracy), ' standard deviation-', np.std(accuracy))
        print('time spent for fitting Naive Bayes Classifier on ', samplesize, 'percent face image for 5 iterations: mean-', 
              sum(speed)/len(speed), 'standard deviation-', np.std(speed))

#Neural Network using TensorFlow with digit data
    trainingdigitimages = [convert_Integer(i).tolist() for i in trainingdigitimages]
    testdigitimages = [convert_Integer(i).tolist() for i in testdigitimages]
    for samplesize in [20,40,60,80,100]:
        sampleimages, samplelabels = select_sample(trainingdigitimages, trainingdigitlabels, int(len(trainingdigitlabels)*samplesize/100))
        model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        model.fit(sampleimages, samplelabels, epochs=5)
        test_loss, test_acc = model.evaluate(testdigitimages, testdigitlabels)
        print('accuracy for Neural Network with Tensorflow on ', samplesize, 'percent digit image for 5 epochs: accuracy-', 
              test_acc, ' loss-', test_loss)
        
    
#Convolutional Neural Network using TensorFlow with face data
    trainingfaceimages = [convert_Integer(i).tolist() for i in trainingfaceimages]
    testfaceimages = [convert_Integer(i).tolist() for i in testfaceimages]
    for samplesize in [20,40,60,80,100]:
        sampleimages, samplelabels = select_sample(trainingfaceimages, trainingfacelabels, int(len(trainingfacelabels)*samplesize/100))
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
        model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
        model.fit(sampleimages, samplelabels, epochs=5)
        test_loss, test_acc = model.evaluate(testfaceimages, testfacelabels)
        print('accuracy for Convolutional Neural Network with Tensorflow on ', samplesize, 'percent face image for 5 epochs: accuracy-', 
              test_acc, ' loss-', test_loss)

if __name__=="__main__":
    main()
