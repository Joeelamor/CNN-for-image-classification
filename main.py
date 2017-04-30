import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import prettytensor as pt
from prettytensor.pretty_tensor_class import PAD_SAME
import input
from input import imgSize, channels, classes

classNames = input.getClassName()
imagesTrain, classTrain, labelTrain = input.loadTraining()
imageTest, classTest, labelTest = input.loadTesting()

print("Training-set:\t\t{}".format(len(imagesTrain)))
print("Test-set:\t\t{}".format(len(imageTest)))


cutImage = 28

# Get the first images from the test-set.
images = imageTest[0:9]

# Get the true classes for those images.
classTrue = classTest[0:9]

x = tf.placeholder(tf.float32, shape=[None, imgSize, imgSize, channels])
y = tf.placeholder(tf.float32, shape=[None, classes])
yClass = tf.argmax(y, dimension=1)

def preProcessImage(image, training):
    if training:
        # Randomly crop the input image.
        image = tf.random_crop(image, size=[cutImage, cutImage, channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:

        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=cutImage,
                                                       target_width=cutImage)
    return image

def preProcess(images, training):

    images = tf.map_fn(lambda image: preProcessImage(image, training), images)

    return images

def CNNetwork(images, training):
    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)

    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases.
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    # Create the convolutional neural network using Pretty Tensor.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        yPred, loss = x_pretty.\
            conv2d(kernel=5, stride=[1, 1, 1, 1], depth=16, batch_normalize=True, edges = PAD_SAME).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, stride=[1, 1, 1, 1], depth=36, edges = PAD_SAME).\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256).\
            fully_connected(size=128).\
            softmax_classifier(num_classes=classes, labels=y)

    return yPred, loss

def createNetwork(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope('network', reuse=not training):
        # Just rename the input placeholder variable for convenience.
        images = x

        # Create TensorFlow graph for pre-processing.
        images = preProcess(images=images, training=training)

        # Create TensorFlow graph for the main processing.
        yPred, loss = CNNetwork(images=images, training=training)

    return yPred, loss

# First create a Tensorflow variable that keeps track of the number of
# optimization iterations performed so far.
trainingStep = tf.Variable(initial_value=0, trainable=False)

# Create the neural network to be used for training. The createNetwork() function
# returns both yPred and loss, but we only need the loss-function during training.
_, loss = createNetwork(training=True)

# Second create the neural network for the test set. Test we only need yPred.
yPred, _ = createNetwork(training=False)

# We calculate the predicted class number as an integer. The output of the network
# yPred is an array with 10 elements. The class number is the index of the largest element in the array.
yPredClass = tf.argmax(yPred, dimension=1)

# We create a vector of booleans telling us whether the predicted class equals the
# true class of each image
correct_prediction = tf.equal(yPredClass, yClass)

# Calculate accuracy
crossEntropy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create an optimizer which will minimize the loss-function. Also pass the
# training_step variable to the optimizer so it will be increased by one after
# each iteration.
optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss, global_step=trainingStep)
# Save variables
saver = tf.train.Saver()

sess = tf.Session()

saveDir = 'checkpoints/'

if not os.path.exists(saveDir):
    os.makedirs(saveDir)

savePath = os.path.join(saveDir, 'cifar10_cnn')

try:
    print("Restore last checkpoint ...")
    lastPath = tf.train.latest_checkpoint(checkpoint_dir=saveDir)
    saver.restore(sess, savePath=lastPath)
    print("Restored checkpoint from:", lastPath)
except:
    print("Initializing variables.")
    sess.run(tf.global_variables_initializer())

train_batch_size = 64

def random_batch():
    num_images = len(imagesTrain)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = imagesTrain[idx, :, :, :]
    y_batch = labelTrain[idx, :]

    return x_batch, y_batch

def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()

        feed_dict_train = {x: x_batch,
                           y: y_true_batch}

        iter, _ = sess.run([trainingStep, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 50 iterations (and last).
        if (iter % 50 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = sess.run(crossEntropy,
                                    feed_dict=feed_dict_train)

            # Print status.
            msg = "Training Step: {0:>6}, Training Batch Accuracy: {1:>.1%}"
            print(msg.format(iter, batch_acc))
            print_test_accuracy()

        # Save a checkpoint to disk every 1000 iterations (and last).
        if (iter % 1000 == 0) or (i == num_iterations - 1):
            saver.save(sess,
                       save_path=savePath,
                       global_step=trainingStep)

            print("Save checkpoint.")
    end_time = time.time()
    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

batch_size = 256

def predictClass(images, labels, classTrue):
    # Number of images.
    num_images = len(images)
    classPred = np.zeros(shape=num_images, dtype=np.int)

    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)
        feed_dict = {x: images[i:j, :], y: labels[i:j, :]}
        classPred[i:j] = sess.run(yPredClass, feed_dict=feed_dict)
        i = j
    # Create a boolean array whether each image is correctly classified.
    correct = (classTrue == classPred)
    return correct, classPred

def predictClassTest():
    return predictClass(images = imageTest, labels = labelTest, classTrue = classTest)

def classification_accuracy(correct):
    return correct.mean(), correct.sum()

def print_test_accuracy():
    correct, classPred = predictClassTest()
    acc, num_correct = classification_accuracy(correct)
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy for Test Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

optimize(num_iterations=10000)