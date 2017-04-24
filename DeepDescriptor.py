import numpy as np
import matplotlib.pyplot as plt
import numpy
import time
from sklearn.decomposition import PCA as sklearnPCA
from numpy import linalg as LA
import sys
import config
import caffe
import logging
import tensorflow as tf
from AutoEncoder import AutoEncoder 

caffe_root = config.CNNInfo['caffe_root']
sys.path.insert(0, caffe_root + 'python')


class DeepDescriptor:

    def __init__(self, layer='pool1', Model='VGG', searchBool = 0, gpu = True):
    	self.layer = layer
        logging.info("instanteating the model")
        self.modeldict = config.LSFModelInfo['model_dict']


        if(config.AutoEncoderInfo['enable_autoencoder'] == 1 and searchBool):
            feature_dim_used = config.CNNInfo['feature_size']
            self.auto_encoder = AutoEncoder(
                config.AutoEncoderInfo['n_layers'], feature_dim_used)
            self.sess = tf.Session(config=self.auto_encoder.config)
            self.X = tf.placeholder("float", [None, feature_dim_used])
            # Construct model
            self.encoder_op = self.auto_encoder.encoder(self.X)
            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.modeldict)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                logging,info("No model found")
        
        caffe.set_device(0)
        if gpu:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        if Model is 'VGG':
            model_def = caffe_root + 'models/vgg/VGG_ave_pool_deploy.prototxt'
            model_weights = caffe_root + 'models/vgg/vgg_normalised.caffemodel'
        elif Model is 'GoogleNet':
            model_def = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
            model_weights = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'

        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)
        mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
        # average over pixels to obtain the mean (BGR) pixel values
        mu = mu.mean(1).mean(1)

        if Model is 'VGG':
            self.transformer = caffe.io.Transformer(
                {'data': self.net.blobs['data'].data.shape})
            self.transformer.set_transpose('data', (2, 0, 1))
            self.transformer.set_mean('data', mu)
            self.transformer.set_raw_scale('data', 255)
            self.transformer.set_channel_swap('data', (2, 1, 0))
            self.InterMediateLayer = layer
        elif Model is 'GoogleNet':

            self.transformer = caffe.io.Transformer(
                {'data': self.net.blobs['data'].data.shape})

            # move image channels to outermost dimension
            self.transformer.set_transpose('data', (2, 0, 1))
            # subtract the dataset-mean value in each channel
            self.transformer.set_mean('data', mu)
            # rescale from [0, 1] to [0, 255]
            self.transformer.set_raw_scale('data', 255)
            # swap channels from RGB to BGR
            self.transformer.set_channel_swap('data', (2, 1, 0))
            self.InterMediateLayer = 'inception_3a/output'
        else:
            loggging.info("Not supported")

    def describe(self, imagePath):
        if config.CNNInfo['keep_aspect_ratio']:
            self.preprocessImage(imagePath)
            image = caffe.io.load_image("tmp.jpg")
        else:
            image = caffe.io.load_image(imagePath)
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image
        output = self.net.forward()
        feat = self.net.blobs[self.InterMediateLayer].data[0]
        shape = feat.shape
        F = np.reshape(feat, [shape[0], shape[1] * shape[2]])
        GramMatrix = np.dot(F, F.T)
        GramMatrix = GramMatrix / LA.norm(GramMatrix)
        GramMatrix = np.around(GramMatrix, decimals=4)
        return GramMatrix

    def search_describe(self, imgPath):
        GramMatrix = self.describe(imgPath);
        input_dim_used = config.CNNInfo['feature_size']
        feature = np.reshape(GramMatrix, (1, input_dim_used))
        batch = np.array(feature)
        if(config.AutoEncoderInfo['enable_autoencoder'] == 1):
            encoded_feature = self.sess.run(
                 self.encoder_op, feed_dict={self.X: batch})
            return encoded_feature
        else:
            return batch

    def preprocessImage(self, imagePath):
        img = cv2.imread(imagePath)
        h, w, _ = img.shape
        if h < w:
            img = skimage.transform.resize(
                img, (256, w * 256 / h), preserve_range=True)
        else:
            img = skimage.transform.resize(
                img, (h * 256 / w, 256), preserve_range=True)
        # Central crop to 224x224
        h, w, _ = img.shape
        img = img[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]
        cv2.imwrite("tmp.jpg", img)
