import numpy as np
import pandas as pd
import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from glob import glob
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from subprocess import check_output
print(check_output(["ls", "./train"]).decode("utf8"))
import tensorflow as tf
sess = tf.InteractiveSession()

TRAIN_DATA = "./train"
TEST_DATA = "./test"
CROP_TRAIN_DATA = "./crop_train"
CROP_TEST_DATA = "./crop_test"

types = ['Type_1','Type_2','Type_3']
type_ids = []

for type in enumerate(types):
    if type[1] != "Test":
        type_i_files = glob(os.path.join(TRAIN_DATA, type[1], "*.jpg"))
        type_i_ids = np.array([s[len(TRAIN_DATA)+8:-4] for s in type_i_files])
    else:
        type_i_files = glob(os.path.join(TEST_DATA, "*.jpg"))
        type_i_ids = np.array([s[len(TEST_DATA)+1:-4] for s in type_i_files])
    type_ids.append(type_i_ids)


def get_cropped_filename(image_id, image_type):
    """
    Method to set cropped image file path from its id and type   
    """
    if image_type == "Type_1" or \
        image_type == "Type_2" or \
        image_type == "Type_3":
        data_path = os.path.join(CROP_TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or \
          image_type == "AType_2" or \
          image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type)
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)
    ext = 'jpg'
    return os.path.join(data_path, "{}-crop.{}".format(image_id, ext))

def load_image_data(image_id, image_type):
    """
    Method to get CROPPED image data as np.array specifying image id and type
    """
    fname = get_cropped_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
 

def mini_batch(images, labels, num):
    assert len(images) == len(labels)
    p = np.random.permutation(len(images))
    images, labels = images[p], labels[p]
    return images[:num], labels[:num]


N = np.sum(len(type) for type in type_ids)  # Number of images in training set
stacked_images = np.empty((N, 256, 256, 3))
labels = np.zeros((N, 3))
count = 0
for type in enumerate(types):
    for i in range(len(type_ids[type[0]])):
        img = load_image_data(type_ids[type[0]][i], type[1])
        stacked_images[count] = img
        labels[count, type[0]] = 1.0
        count += 1

stacked_images = stacked_images.reshape(-1,256*256*3)

x = tf.placeholder(tf.float32, shape=[None, 256*256*3])
y_ = tf.placeholder(tf.float32, shape=[None, 3])  # 3 distinct classes 

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Start building computational graph for ConvNN... 

# First layer

W_conv1 = weight_variable([5, 5, 3, 16])
b_conv1 = bias_variable([16])

x_image = tf.reshape(x, [-1, 256, 256, 3])  # 256 x 256 pixels, 3 colour channels

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# Second layer

W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Densely connected layer

W_fc1 = weight_variable([32 * 32 * 32, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(1000):
    x_batch, y_batch = mini_batch(stacked_images, labels, 1)
    print("###################")
    print("x_batch shape is {}, y_batch shape is {}".format(x_batch.shape, y_batch.shape))
    print("###################")
    if i%100 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x_image: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))