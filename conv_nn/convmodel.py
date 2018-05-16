# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3
batch_size = 4

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), 3)  # [float(i)] [one_hot(float(i), num_classes)]
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch

"""
filenames0 = tf.train.match_filenames_once("dataset/airplane_train/*.jpg")
filenames1 = tf.train.match_filenames_once("dataset/motorcycle_train/*.jpg")
filenames2 = tf.train.match_filenames_once("dataset/watch_train/*.jpg")

filenames3 = tf.train.match_filenames_once("dataset/airplane_validation/*.jpg")
filenames4 = tf.train.match_filenames_once("dataset/motorcycle_validation/*.jpg")
filenames5 = tf.train.match_filenames_once("dataset/watch_validation/*.jpg")

filename_queue0 = tf.train.string_input_producer(filenames0, shuffle=False)
filename_queue1 = tf.train.string_input_producer(filenames1, shuffle=False)
filename_queue2 = tf.train.string_input_producer(filenames2, shuffle=False)

filename_queue3 = tf.train.string_input_producer(filenames3, shuffle=False)
filename_queue4 = tf.train.string_input_producer(filenames4, shuffle=False)
filename_queue5 = tf.train.string_input_producer(filenames5, shuffle=False)

reader0 = tf.WholeFileReader()
reader1 = tf.WholeFileReader()
reader2 = tf.WholeFileReader()

reader3 = tf.WholeFileReader()
reader4 = tf.WholeFileReader()
reader5 = tf.WholeFileReader()

key0, file_image0 = reader0.read(filename_queue0)
key1, file_image1 = reader1.read(filename_queue1)
key2, file_image2 = reader2.read(filename_queue2)

key3, file_image3 = reader3.read(filename_queue3)
key4, file_image4 = reader4.read(filename_queue4)
key5, file_image5 = reader5.read(filename_queue5)

image0, label0 = tf.image.decode_jpeg(file_image0), [1., 0., 0.]  # key0
image0 = tf.reshape(image0, [80, 140, 1])

image1, label1 = tf.image.decode_jpeg(file_image1), [0., 1., 0.]  # key1
image1 = tf.reshape(image1, [80, 140, 1])

image2, label2 = tf.image.decode_jpeg(file_image2), [0., 0., 1.]  # key2
image2 = tf.reshape(image2, [80, 140, 1])

image3, label3 = tf.image.decode_jpeg(file_image3), [1., 0., 0.]   # key3
image3 = tf.reshape(image3, [80, 140, 1])

image4, label4 = tf.image.decode_jpeg(file_image4), [0., 1., 0.]   # key4
image4 = tf.reshape(image4, [80, 140, 1])

image5, label5 = tf.image.decode_jpeg(file_image5), [0., 0., 1.]   # key5
image5 = tf.reshape(image5, [80, 140, 1])

image0 = tf.to_float(image0) / 256. - 0.5
image1 = tf.to_float(image1) / 256. - 0.5
image2 = tf.to_float(image2) / 256. - 0.5

image3 = tf.to_float(image3) / 256. - 0.5
image4 = tf.to_float(image4) / 256. - 0.5
image5 = tf.to_float(image5) / 256. - 0.5

batch_size = 4

min_after_dequeue = 10  # 10000
capacity = min_after_dequeue + 3 * batch_size


example_batch0, label_batch0 = tf.train.shuffle_batch([image0, label0], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

example_batch1, label_batch1 = tf.train.shuffle_batch([image1, label1], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

example_batch2, label_batch2 = tf.train.shuffle_batch([image2, label2], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

valid_set0, valid_label0 = tf.train.shuffle_batch([image3, label3], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

valid_set1, valid_label1 = tf.train.shuffle_batch([image4, label4], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

valid_set2, valid_label2 = tf.train.shuffle_batch([image5, label5], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

example_batch = tf.concat(values=[example_batch0, example_batch1, example_batch2], axis=0)
label_batch = tf.concat(values=[label_batch0, label_batch1, label_batch2], axis=0)

valid_set = tf.concat(values=[valid_set0, valid_set1, valid_set2], axis=0)
valid_label = tf.concat(values=[valid_label0, valid_label1, valid_label2], axis=0)
"""


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(["dataset/airplane_train/*.jpg", "dataset/motorcycle_train/*.jpg",
                                                     "dataset/watch_train/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["dataset/airplane_validation/*.jpg", "dataset/motorcycle_validation/*.jpg",
                                                     "dataset/watch_validation/*.jpg"], batch_size=batch_size)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, dtype=tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, dtype=tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

"""
o1 = tf.layers.conv2d(inputs=example_batch, filters=32, kernel_size=3, activation=tf.nn.relu)
o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)

cost = tf.reduce_sum(tf.square(y - label_batch))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
"""


# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()
array_train = []
array_valid = []

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for _ in range(500):
        # a, c = sess.run([optimizer, cost], feed_dict={x: valid_set, y: valid_label})
        a, c = sess.run([optimizer, cost])
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(label_batch_train))
            sess.run(label_batch_valid)

            error_train = sess.run(cost)
            array_train.append(error_train)
            print(error_train)
            error_valid = sess.run(cost_valid)
            array_valid.append(error_valid)
            print(error_valid)

            print(sess.run(example_batch_valid_predicted))
            print("Error:", sess.run(cost_valid))

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)


plt_err_train, = plt.plot(array_train, label='Error training')
plt_err_valid, = plt.plot(array_valid, label='Error valid')

plt.legend(handles=[plt_err_train, plt_err_valid])
plt.xlabel("epoch")
plt.ylabel("error")
plt.show()
