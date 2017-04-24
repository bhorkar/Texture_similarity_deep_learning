from __future__ import print_function
import numpy as np
import tensorflow as tf
import math as math
import glob
import cv2
import optparse
import os
import matplotlib.pyplot as plt
import skimage.transform
import config
import logging
import pickle



class AutoEncoder(object):

    def encoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        if self.n_layer == 1:
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
            return layer_1
        elif self.n_layer == 2:
        # Decoder Hidden layer with sigmoid activation #2
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
            return layer_2

    def decoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        if self.n_layer == 1:
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h2']),
                                           self.biases['decoder_b2']))
            return layer_1
        elif self.n_layer == 2:
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                           self.biases['decoder_b1']))
            # Decoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                           self.biases['decoder_b2']))
            return layer_2

    def __init__(self, n_layers, feature_dim_used):
        learning_rate = 0.001
        self.batch_size = 16
        self.display_step = 100
        self.n_layer = n_layers;
        self.n_hidden_1 = 512  # 1st layer num features
        self.n_hidden_2 = 64  # 2nd layer num features
        n_hidden_1 = self.n_hidden_1; 
        n_hidden_2 = self.n_hidden_2;
        n_input = feature_dim_used
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        self.config = tf.ConfigProto(gpu_options=gpu_options)

        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="eh1"),
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="eh2"),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]), name="dh1"),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]), name="dh2"),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1]), name="eb1"),
            'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2]), name="eb2"),
            'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1]), name="db2"),
            'decoder_b2': tf.Variable(tf.random_normal([n_input]), name="db2"),
        }
        self.X = tf.placeholder("float", [None, feature_dim_used])
        self.encoder_op = self.encoder(self.X)
        self.decoder_op = self.decoder(self.encoder_op)
        self.y_pred = self.decoder_op
        self.y_true = self.X


# Define loss and optimizer, minimize the squared error
        self.cost = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate).minimize(self.cost)


if __name__ == '__main__':
    pass
