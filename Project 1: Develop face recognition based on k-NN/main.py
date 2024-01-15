from os import listdir
from os.path import isdir
from numpy import asarray
import numpy as np
import cv2 as cv
from sklearn.preprocessing import LabelEncoder
from math import sqrt

width = 128 # image width
height = 128 # image height 
dim = (width, height)


def load_images(directory):
    images = list()
    for filename in listdir(directory):
        path = directory + filename # path
        img = cv.imread(path) 
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA) # resize
        normalized_image = cv.normalize(resized, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
        arr = np.array(normalized_image) # convert to numpy array
        newarr = arr.reshape(-1) # convert to 1-D array
        images.append(newarr) # store

    return images

# load a dataset
def load_dataset(directory):
    x, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/' # path
        if not isdir(path):
            continue
        images = load_images(path)
        labels = [subdir for _ in range(len(images))] # generate label list
        print('>loaded %d examples for class: %s' % (len(images), subdir))
        print('Label name:', labels)
        x.extend(images) # add images to X
        y.extend(labels) # add label to Y

    return asarray(x), asarray(y)


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)

# Locate the most similar neighbors 
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

# make a classiciation prediction with neighbors 
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


trainX, trainY = load_dataset('dataset/images/')
le = LabelEncoder()
label = le.fit_transform(trainY) # categorical encoding 
labelCol = label[:, np.newaxis] # convert to column vector 
print(trainX.shape)
#print(labelCol)
data = np.concatenate((trainX, labelCol), axis=1)
print(data.shape)

# load unknown image 
img = cv.imread('jackie-chan.jpg')
resized = cv.resize(img, dim, interpolation=cv.INTER_AREA) #resize 
normalized_image = cv.normalize(resized, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
arr = np.array(normalized_image) # convert to numpy array
imgUnknown = arr.reshape(-1) # convert to 1-D array

expect = 0
prediction = predict_classification(trainX, imgUnknown, 3)
print('\n---------\nExpected %d, Predicted %d.' % (expect, prediction))
