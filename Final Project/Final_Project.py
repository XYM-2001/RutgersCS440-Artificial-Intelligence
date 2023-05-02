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
    traininglabels = load_label('digitdata\\traininglabels')
    traininglabels = [int(i) for i in traininglabels]
    trainingimages = load_image('digitdata\\trainingimages', len(traininglabels), 28, 28)
    testlabels = load_label('digitdata\\testlabels')
    testlabels = [int(i) for i in testlabels]
    testimages = load_image('digitdata\\testimages', len(testlabels), 28, 28)
    perceptrons = []
    for i in range(10):
        perceptrons.append(Perceptron(28*28))
    perceptrons = train_digit_model(trainingimages, traininglabels, perceptrons)
    trues = 0
    for i in range(len(testimages)):
        features = convert_Integer(testimages[i]).flatten()
        predictions = []
        for j in range(10):
            predictions.append(perceptrons[j].predict(features))
        prediction = predictions.index(max(predictions))
        if prediction == testlabels[i]:
            trues += 1
    print('accuracy for Perceptron on testing digits: ', trues/len(testlabels))

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
    # print('accuracy for Perceptron on testing faces: ', trues/len(testlabels))


#Naive Bayes for digit data
    # traininglabels = load_label('digitdata\\traininglabels')
    # traininglabels = [int(i) for i in traininglabels]
    # trainingimages = load_image('digitdata\\trainingimages', len(traininglabels), 28, 28)
    # testlabels = load_label('digitdata\\testlabels')
    # testlabels = [int(i) for i in testlabels]
    # testimages = load_image('digitdata\\testimages', len(testlabels), 28, 28)
    # trainingclasses = image_split(trainingimages, traininglabels,10)
    # classifier = []
    # for i in range(10):
    #     classifier.append(NaiveBayes(28*28,len(trainingclasses[i])/len(trainingimages)))
    # for i in range(10):
    #     classifier[i].fit(trainingclasses[i])
    # trues = 0
    # for i in range(len(testimages)):
    #     features = convert_Integer(testimages[i]).flatten()
    #     predictions = []
    #     for j in range(10):
    #         predictions.append(classifier[j].predict(features))
    #     prediction = predictions.index(max(predictions))
    #     if prediction == testlabels[i]:
    #         trues += 1
    # print('accuracy for Perceptron on testing digits: ', trues/len(testlabels))

#Naive Bayes for face data
    # traininglabels = load_label('facedata\\facedatatrainlabels')
    # traininglabels = [int(i) for i in traininglabels]
    # trainingimages = load_image('facedata\\facedatatrain', len(traininglabels), 60, 70)
    # testlabels = load_label('facedata\\facedatatestlabels')
    # testlabels = [int(i) for i in testlabels]
    # testimages = load_image('facedata\\facedatatest', len(testlabels), 60, 70)
    # trainingclasses = image_split(trainingimages,traininglabels,2)
    # classifier = []
    # for i in range(2):
    #     classifier.append(NaiveBayes(60*70,len(trainingclasses[i])/len(trainingimages)))
    # for i in range(2):
    #     classifier[i].fit(trainingclasses[i])
    # trues = 0
    # for i in range(len(testimages)):
    #     predictions = []
    #     predictions.append(classifier[0].predict(convert_Integer(testimages[i]).flatten()))
    #     predictions.append(classifier[1].predict(convert_Integer(testimages[i]).flatten()))
    #     prediction = predictions.index(max(predictions))
    #     if prediction == testlabels[i]:
    #         trues += 1
    # print('accuracy for naive bayes on testing faces: ', trues/len(testlabels))

#Neural Network using TensorFlow with digit data
    # traininglabels = load_label('digitdata\\traininglabels')
    # traininglabels = [int(i) for i in traininglabels]
    # trainingimages = load_image('digitdata\\trainingimages', len(traininglabels), 28, 28)
    # trainingimages = [convert_Integer(i).tolist() for i in trainingimages]
    # testlabels = load_label('digitdata\\testlabels')
    # testlabels = [int(i) for i in testlabels]
    # testimages = load_image('digitdata\\testimages', len(testlabels), 28, 28)
    # testimages = [convert_Integer(i).tolist() for i in testimages]
    # model = tf.keras.Sequential([
    # tf.keras.layers.Flatten(input_shape=(28, 28)),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(10, activation='softmax')
    # ])
    # model.compile(optimizer='adam',
    #           loss='sparse_categorical_crossentropy',
    #           metrics=['accuracy'])
    # model.fit(trainingimages, traininglabels, epochs=10)
    # test_loss, test_acc = model.evaluate(testimages, testlabels)
    # print('Neural Nework digit data Test accuracy:', test_acc, 'Test loss:', test_loss)
    
#Convolutional Neural Network using TensorFlow with face data
    # traininglabels = load_label('facedata\\facedatatrainlabels')
    # traininglabels = [int(i) for i in traininglabels]
    # trainingimages = load_image('facedata\\facedatatrain', len(traininglabels), 60, 70)
    # trainingimages = [convert_Integer(i).tolist() for i in trainingimages]
    # testlabels = load_label('facedata\\facedatatestlabels')
    # testlabels = [int(i) for i in testlabels]
    # testimages = load_image('facedata\\facedatatest', len(testlabels), 60, 70)
    # testimages = [convert_Integer(i).tolist() for i in testimages]
    # model = tf.keras.Sequential([
    # tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(70, 60, 1)),
    # tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(1, activation='sigmoid')
    # ])
    # model.compile(optimizer='adam',
    #           loss='binary_crossentropy',
    #           metrics=['accuracy'])
    # model.fit(trainingimages, traininglabels, epochs=10)
    # test_loss, test_acc = model.evaluate(testimages, testlabels)
    # print('Neural Nework facedata Test accuracy:', test_acc, 'Test loss:', test_loss)

if __name__=="__main__":
    main()
