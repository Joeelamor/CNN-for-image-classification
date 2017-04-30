import numpy as np
import pickle
import os

data_path = "data/"

# Input image's size 32 * 32
imgSize = 32

# Number of channels in each image, 3 channels: red, green, blue
channels = 3

# Number of classes
classes = 10

# Number of files for the training batches
numOfFiles = 5

# Number of images in file in the training-set
numImage = 10000

# Total number of images in the training-set
totalTrain = numOfFiles * numImage

# One-hot encode sample labels
# num : list of sample lables, return array of one-hot encode labels
def one_hot(num, classes = None):
    if classes is None:
        classes = np.max(classes) - 1

    return np.eye(classes)[num]

# Get the file path
def getPath(filename = ""):
    return os.path.join(data_path, 'cifar-10-batches-py/', filename)

# Unpickle file
def unpickle(filename):
    filePath = getPath(filename)

    with open(filePath, 'rb') as file:
        dict = pickle.load(file, encoding = 'bytes')
    return dict

# Reshape images into [image_number, height, width, channel]
def convert(oriImg):
    raw_float = np.array(oriImg, dtype=float) / 255.0
    images = raw_float.reshape([-1, channels, imgSize, imgSize])
    images = images.transpose([0, 2, 3, 1])

    return images

# Load file
def loadFile(filename):
    # unpickle file
    file = unpickle(filename)

    # get the original image
    oriImg = file[b'data']

    # get the class for every image
    label = np.array(file[b'labels'])

    images = convert(oriImg)

    return images, label

# Get class names
def getClassName():

    # unpickle the pickled file
    label = unpickle(filename = 'batches.meta')[b'label_names']

    # convert from binary strings
    name = [x.decode('utf-8') for x in label]

    return name

# Load training data
def loadTraining():
    images = np.zeros(shape = [totalTrain, imgSize, imgSize, channels], dtype = float)
    labels = np.zeros(shape = [totalTrain], dtype = int)

    currentIndex = 0

    # Merge all the batch file
    for i in range(numOfFiles):
        imgBatch, labelBatch = loadFile(filename = "data_batch_" + str(i + 1))

        numImageBatch = len(imgBatch)
        nextIndex = currentIndex + numImageBatch

        # store image and label in the array
        images[currentIndex:nextIndex, :] = imgBatch
        labels[currentIndex:nextIndex] = labelBatch

        currentIndex = nextIndex

    return images, labels, one_hot(num = labels, classes = classes)

def loadTesting():
    images, labels = loadFile(filename = 'test_batch')
    return images, labels, one_hot(num = labels, classes = classes)