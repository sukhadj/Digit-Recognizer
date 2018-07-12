""" This file contains the model function for each model """
from typing import Dict, Any

import mnist
import numpy as np
import tensorflow as tf
from tensorflow import layers

tf.logging.set_verbosity(tf.logging.INFO)


def download():
    print("================= Downloading the data ===============")
    # Download the train images and train label
    train_images = mnist.train_images()
    train_labels = np.asarray(mnist.train_labels(), np.int32)
    test_imgaes = mnist.test_images()
    test_labels = np.asarray(mnist.test_labels(), np.int32)
    return train_images, train_labels, test_imgaes, test_labels


def normalize(images):
    # Normalize the input between range 0 to 1
    images = images / 255.0
    images = images.astype(np.float32)
    return images


def cnn_model(features, labels, mode):
    """ CNN MODEL"""
    """ ((Conv2D)*2-->MaxPool-->Dropout)*2-->Dense-->dropout-->softmax """

    # input layer
    input_layer = tf.reshape(features["x"], (-1, 28, 28, 1))

    # conv1
    conv1 = layers.conv2d(inputs=input_layer,
                          filters=32,
                          kernel_size=(5, 5),
                          padding="same",
                          data_format="channels_last",
                          activation=tf.nn.relu)

    # conv2
    conv2 = layers.conv2d(inputs=conv1,
                          filters=32,
                          kernel_size=(5, 5),
                          padding="same",
                          data_format="channels_last",
                          activation=tf.nn.relu)

    # max1
    max1 = layers.max_pooling2d(inputs=conv2,
                                pool_size=(2, 2),
                                strides=2)

    # drop1
    drop1 = layers.dropout(inputs=max1, rate=0.25, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # conv3
    conv3 = layers.conv2d(inputs=drop1,
                          filters=64,
                          kernel_size=(3, 3),
                          padding="same",
                          data_format="channels_last",
                          activation=tf.nn.relu)

    # conv4
    conv4 = layers.conv2d(inputs=conv3,
                          filters=64,
                          kernel_size=(3, 3),
                          padding="same",
                          data_format="channels_last",
                          activation=tf.nn.relu)

    # max1
    max2 = layers.max_pooling2d(inputs=conv4,
                                pool_size=(2, 2),
                                strides=2)

    # drop1
    drop2 = layers.dropout(inputs=max2, rate=0.25, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Dense layer
    flat1 = tf.reshape(drop2, (-1, 7 * 7 * 64))
    dense1 = layers.dense(inputs=flat1, units=256, activation=tf.nn.relu)
    drop3 = layers.dropout(inputs=dense1, rate=0.5)

    logits = layers.dense(inputs=drop3, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits=logits, name="softmax_tensor"),

    }  # type: Dict[str, Any]

    # if mode is eval of predict return this dict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate the loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10),
                                           logits=logits)

    # Optimize for training process
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step()
                                      )

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def neural_network(features, labels, mode):
    """  A Simple neural network """
    """ 28*28-->1024-->128-->10 """
    input_layer = tf.reshape(features["x"], [-1, 28*28])

    dense1 = tf.layers.dense(inputs=input_layer, units=1024, activation=tf.nn.relu, name="dense1")
    dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu, name="dense2")
    drop1 = tf.layers.dropout(inputs=dense2, rate=0.5, training=(mode==tf.estimator.ModeKeys.TRAIN), name="drop1")

    logits = tf.layers.dense(inputs=drop1, units=10, name="logits")

    predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "predictions": tf.nn.softmax(logits=logits, name="softmax_tensor")
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(indices=tf.cast(labels, tf.uint8), depth=10),
                                           logits=logits)

    eval_metric = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op, eval_metric)

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,loss=loss, eval_metric_ops=eval_metric)
