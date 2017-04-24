from __future__ import print_function
import numpy as np
import tensorflow as tf
import math as math
import glob
import cv2
from DeepDescriptor import DeepDescriptor
import optparse
import os
import matplotlib.pyplot as plt
import skimage.transform
import config
import logging
from sklearn.neighbors import LSHForest
import pickle
from AutoEncoder import AutoEncoder


class CnnActivation(object):

    def file_len(self, Filelist_glob):
        flen = 0
        for imagePath in Filelist_glob:
            flen = flen + 1
        return flen

    def __init__(self, descriptor, model_dict,
                 feature_dim_used, trainImageDict, indexFile):
        self.train_path = trainImageDict
        self.model_dict = model_dict
        self.cd = descriptor
        self.indexFile = indexFile
        self.feature_dim_used = feature_dim_used
        self.flen = self.file_len(glob.iglob(self.train_path + "/*.jpg"))
        self.Filelist = glob.iglob(self.train_path + "/*.jpg")

    def readfileFromdict(self, Filelist, batch_size=1):
        n = 0
        read_features = np.array([], dtype=np.float).reshape(
            0, self.feature_dim_used)
        imageset = []
        for imagePath in Filelist:
            # path and load the image itself
            imageID = imagePath[imagePath.rfind("/") + 1:]
            imageset.append(imageID)
            features = self.cd.describe(imagePath)
            features = np.asarray(features.ravel()).reshape(1, -1)[0][:]
            features = features[:self.feature_dim_used]
            read_features = np.vstack((read_features, features))
            if(n == batch_size - 1):
                break
            n = n + 1
        return read_features, imageset

    def EncodeTextureLSFTree(self, ntrees, autoencoderEnable):
     # Restore variables from disk.
        if autoencoderEnable == True:
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess = tf.Session()
            ckpt = tf.train.get_checkpoint_state(self.model_dict)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                logging.info("No model found")
                return
            logging.info("Tensorflow Model restored.")
        # store data in number of batches
        nSamplesPerTree = int(math.floor(self.flen / ntrees))
        for nt in xrange(ntrees):
            image_idx = open(self.indexFile + str(nt), "w")
            logging.info("Inside " + str(nt) + " tree")
            if autoencoderEnable == True:
                if self.auto_encoder.n_layer == 1:
                    X_train = np.array([], dtype=np.float).reshape(
                        0, self.auto_encoder.n_hidden_1)
                elif self.auto_encoder.n_layer == 2:
                    X_train = np.array([], dtype=np.float).reshape(
                        0, self.auto_encoder.n_hidden_2)
            else:
                X_train = np.array([], dtype=np.float).reshape(
                    0, self.feature_dim_used)
            for i in xrange(nSamplesPerTree):
                batch = []
                if i % 1000 == 1:
                    logging.info("Processed:" + str(i))
                batch, imagename = self.readfileFromdict(self.Filelist)
                image_idx.write("%s \n" % imagename)
                batch = np.array(batch)
                if autoencoderEnable == True:
                    image_idx.write("%s \n" % imagename)
                    batch = np.array(batch)
                    encode = sess.run(self.auto_encoder.encoder_op, feed_dict={
                                      self.auto_encoder.X: batch})
                    X_train = np.vstack((X_train, encode))
                else:
                    X_train = np.vstack((X_train, batch))
            logging.info(X_train.shape)
            image_idx.close()
            logging.info(str(X_train.shape))
            lshf = LSHForest(config.LSFModelInfo[
                             'n_candidates'], config.LSFModelInfo['n_estimators'])
            lshf.fit(X_train)
            pickle.dump(lshf, open(self.model_dict +
                                   "LSFModelsave.p" + str(nt), "wb"))

    def runAutoEncoder(self, auto_encoder):
        self.auto_encoder = auto_encoder
        logging.info("Running the ecoder")
        init = tf.global_variables_initializer()
        saver2 = tf.train.Saver()
        AutoEncoderFilelist = glob.iglob(self.train_path + "/*.jpg")
# Launch the graph
        with tf.Session(config=auto_encoder.config) as sess:
            sess.run(init)
            total_batch = int(self.flen / auto_encoder.batch_size)

            for epoch in range(total_batch):
                batch_xs, image = self.readfileFromdict(
                    AutoEncoderFilelist, auto_encoder.batch_size)
                # Run optimization op (backprop) and cost op (to get loss
                # value)
                _, c = sess.run([auto_encoder.optimizer, auto_encoder.cost],
                                feed_dict={auto_encoder.X: np.array(batch_xs)})
            # Display logs per epoch step
                if epoch % auto_encoder.display_step == 0:
                    print("Epoch:", '%04d' %
                          (epoch + 1), "cost=", "{:.9f}".format(c))
                    save_path = saver2.save(
                        sess, self.model_dict + "model1.ckpt", global_step=epoch)

            logging.info("Optimization Finished!")
            logging.info("Model2 saved in file: %s" % save_path)


class TextureEncode(object):
    default_args = {
        # update with model directories
    }

    def __init__(self, gpu_mode):
        indexFile = config.LSFModelInfo[
            'model_dict'] + config.LSFModelInfo['indexFile']
        ntrees = config.LSFModelInfo['nLSFModelTrees']
        feature_dim_used = config.CNNInfo['feature_size']
        model_dict = config.LSFModelInfo['model_dict']
        net = config.CNNInfo['net']
        layer = config.CNNInfo['layer']
        train_path = config.CNNInfo['input_dict']
        if not os.path.exists(model_dict):
            os.makedirs(model_dict)
        logging.info('Loading net and associated files...')
        if config.AutoEncoderInfo['enable_autoencoder']:
            # hack. tensorflow may not work caffee otherwise. initilize tensor
            # before caffe
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.cnn_activation = CnnActivation(DeepDescriptor(layer, net, gpu = gpu_mode),
                                            model_dict, feature_dim_used, train_path, indexFile)
        if config.AutoEncoderInfo['enable_autoencoder']:
            self.autoencoder = AutoEncoder(
                config.AutoEncoderInfo['n_layers'], feature_dim_used)
            self.cnn_activation.runAutoEncoder(self.autoencoder)
        self.cnn_activation.EncodeTextureLSFTree(
            ntrees, config.AutoEncoderInfo['enable_autoencoder'])
        # load the result image and display

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=True)

    opts, args = parser.parse_args()
    TextureEncode.default_args.update({'gpu_mode': opts.gpu})
    ted = TextureEncode(**TextureEncode.default_args)
    logging.info("Finished Feature extraction")
