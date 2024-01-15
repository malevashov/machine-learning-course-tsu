from os import listdir
from os.path import isdir
from numpy import asarray
import numpy as np
import cv2 as cv
from sklearn.preprocessing import LabelEncoder
from math import sqrt

width = 128
height = 128
dim = (width, height)


def load_images(directory):
    images = list()
    for filename in listdir(directory):
        path = directory + filename
        img = cv.imread(path)
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        normalized_image = cv.normalize(resized, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
        arr = np.array(normalized_image)
        newarr = arr.reshape(-1)
        images.append(newarr)

    return images


def load_dataset(directory):
    x, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        if not isdir(path):
            continue
        images = load_images(path)
        labels = [subdir for _ in range(len(images))]
        print('>loaded %d examples for class: %s' % (len(images), subdir))
        print('Label name:', labels)
        x.extend(images)
        y.extend(labels)

    return asarray(x), asarray(y)


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


trainX, trainY = load_dataset('dataset/images/')
le = LabelEncoder()
label = le.fit_transform(trainY)
labelCol = label[:, np.newaxis]
print(trainX.shape)
#print(labelCol)
data = np.concatenate((trainX, labelCol), axis=1)
print(data.shape)

img = cv.imread('jackie-chan.jpg')
resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
normalized_image = cv.normalize(resized, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
arr = np.array(normalized_image)
imgUnknown = arr.reshape(-1)

expect = 0
prediction = predict_classification(trainX, imgUnknown, 3)
print('\n---------\nExpected %d, Predicted %d.' % (expect, prediction))
