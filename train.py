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
CROP_DATA = "./crop_train"

types = ['Type_1','Type_2','Type_3']
type_ids = []

for type in enumerate(types):
    type_i_files = glob(os.path.join(TRAIN_DATA, type[1], "*.jpg"))
    type_i_ids = np.array([s[len(TRAIN_DATA)+8:-4] for s in type_i_files])
    type_ids.append(type_i_ids)


def get_cropped_filename(image_id, image_type):
    """
    Method to set cropped image file path from its id and type   
    """
    if image_type == "Type_1" or \
        image_type == "Type_2" or \
        image_type == "Type_3":
        data_path = os.path.join(CROP_DATA, image_type)
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
 

def mini_batch(array, num):
	pass

N = np.sum(len(type) for type in type_ids)  # Number of images in training set
DATA = np.empty((N, 255, 255, 3), dtype=np.float32)
LABELS = np.zeros((N, 3), dtype=np.float32)
count = 0
for type in enumerate(types):
	for i in range(len(type_ids[type[0]])):
 		img = load_image_data(type_ids[type[0]][i], type[1])
		print("img shape is:", img.shape)
		DATA[count, :, :, :] = img
		LABELS[count, type[0]] = 1.0
		count += 1



x_image = tf.placeholder(tf.float32, shape=[None, 255, 255, 3])  # 255 x 255 pixels, 3 colour channels
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

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# Second layer

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Densely connected layer

W_fc1 = weight_variable([64 * 64 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
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
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))





# for j in range(1):
# 	for i in range(1):
#  		img = load_image_data(type_ids[type[j]][i], type[1])
# 		print("img is: ", img)
# 		print("img shape is:", img.shape)