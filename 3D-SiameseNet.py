"""
<3D-Siamese Net architecture, data loading, and training>
    Copyright (C) <2019>  <Cecilia Ostertag>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import keras
from keras import Sequential, Model
from keras.layers import Conv3D, AveragePooling3D, BatchNormalization, ZeroPadding3D, Dropout, Activation, Flatten, Dense, Input, concatenate, Lambda
from keras import models
from keras import backend as K
import tensorflow as tf
import os
import sys
import gc
from matplotlib import pyplot as plt
import numpy as np
import cv2
import argparse
import nibabel as nib
import skimage
import scipy
from scipy.ndimage import zoom
from random import shuffle
import glob
from keras.utils.training_utils import multi_gpu_model
from keras import optimizers

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
###### TF RECORDS UTILITY FUNCTIONS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def _read_from_tfrecord(example_proto):
    feature = {
        'train/label': tf.FixedLenFeature([], tf.int64),
        'train/image': tf.FixedLenFeature([], tf.string),
        'train/image2': tf.FixedLenFeature([], tf.string)
    }

    features = tf.parse_example([example_proto], features=feature)

    label_1d = features['train/label']
    image_1d = tf.decode_raw(features['train/image'], tf.float32)
    image2_1d = tf.decode_raw(features['train/image2'], tf.float32)

    label_restored = label_1d
    zmax = 150
    xmax = 205
    ymax = 216
    xmax = xmax//2
    ymax = ymax//2
    zmax = zmax//2
    image_restored = tf.reshape(image_1d, [xmax, ymax, zmax])
    image2_restored = tf.reshape(image2_1d, [xmax, ymax, zmax])
    return label_restored, image_restored, image2_restored
    
###### Utility functions

def plotExampleImage(image,title):
	fig = plt.figure(figsize=(10,2))
	plt.title(title)
	cols = 3
	rows = 1
	volume = image.reshape(image.shape[0],image.shape[1],image.shape[2])
	proj0 = np.mean(volume, axis=0)
	proj1 = np.mean(volume, axis=1)
	proj2 = np.mean(volume, axis=2)
	ax1 = fig.add_subplot(rows, cols, 1)
	ax1.title.set_text("axis 0")
	plt.imshow(proj0,cmap="gray") 
	ax2 = fig.add_subplot(rows, cols, 2)
	ax2.title.set_text("axis 1")
	plt.imshow(proj1,cmap="gray")
	ax3 = fig.add_subplot(rows, cols, 3)
	ax3.title.set_text("axis 2")
	plt.imshow(proj2,cmap="gray")
	
def saveExampleImage(image,title):
	volume = image.reshape(image.shape[0],image.shape[1],image.shape[2])
	proj0 = np.mean(volume, axis=0)
	cv2.imwrite(title+".jpg",proj0)
    
class DataGenerator(keras.utils.Sequence):
	def __init__(self, dataset, next1, sess, nbsamples, batchsize, augment=False):
		self.dataset = dataset
		self.next1 = next1
		self.sess = sess
		self.nbsamples = nbsamples
		self.batchsize = batchsize
		self.augment = augment
		self.i = 1

	def __len__(self):
		'Denotes the number of batches per epoch'
		return self.nbsamples // self.batchsize

	def __getitem__(self, index):
		'Generate one batch of data'
		zmax = 150
		xmax = 205
		ymax = 216
		xmax = xmax//2
		ymax = ymax//2
		zmax = zmax//2
		batch = self.sess.run(self.next1)
		imgs = batch[1]
		imgs2 = batch[2]

		if self.augment == True:
			for i in range(imgs.shape[0]):
				angle = np.random.randint(0, 6)
				neg = np.random.randint(0,2)
				if neg == 1:
					angle = - angle
				flip = np.random.randint(0,2)
				imgs[i,:,:,:] = scipy.ndimage.interpolation.rotate(imgs[i,:,:,:], angle, axes=(1,2), reshape=False)
				imgs2[i,:,:,:] = scipy.ndimage.interpolation.rotate(imgs2[i,:,:,:], angle, axes=(1,2), reshape=False)
				if flip == 1:
					imgs[i,:,:,:] = np.flip(imgs[i,:,:,:], axis=2)	
					imgs2[i,:,:,:] = np.flip(imgs2[i,:,:,:], axis=2)		
		imgs = imgs.reshape((-1, 1, xmax, ymax, zmax))
		imgs2 = imgs2.reshape((-1, 1, xmax, ymax, zmax))
		labels = batch[0].reshape((-1,))
		print(self.i)
		if self.augment == False:
			print(labels)
		self.i += 1
		return [imgs,imgs2], keras.utils.to_categorical(labels, num_classes=2)


#### Load data and crate Dataset objects
train_filename = 'train_siamese_resize.tfrecords'
epochs=800
batch_size = 20 # erreur de segmentation avec un batch size trop grand sur cpu
data_path = tf.placeholder(dtype=tf.string, name="tfrecord_file")
#print(train_filename.dtype)

for i in range(10): # 10 times random sub-sampling cross-validation
	print("Test "+str(i+1))

	dataset = tf.data.TFRecordDataset(data_path)
	dataset = dataset.map(_read_from_tfrecord).shuffle(256,None,False)
	dataval = dataset.take(40)
	dataval = dataval.repeat().batch(batch_size) #don't shuffle validation data ?
	datatrain = dataset.skip(40)
	datatrain = datatrain.shuffle(buffer_size=216).repeat().batch(batch_size)

	gc.collect()

	##########################


	nb_classes=2
	
	#####

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	K.set_session(sess)
	K.set_image_data_format("channels_first")
	zmax = 150
	xmax = 205
	ymax = 216
	xmax = xmax//2
	ymax = ymax//2
	zmax = zmax//2

	reg = 0.005 #L2 regularization factor

	with tf.device('/cpu:0'): # Create Siamese network architecture
		left_input = Input((1,xmax,ymax,zmax),name="left_input")
		right_input = Input((1,xmax,ymax,zmax),name="right_input")
		inputs = Input((1,xmax,ymax,zmax))
	
		x = BatchNormalization(axis=1, momentum=0.99)(inputs)

		c1 = Conv3D(16, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)
		x = BatchNormalization(axis=1, momentum=0.99)(c1)
		x = keras.layers.LeakyReLU(alpha=0.01)(x)
		x = AveragePooling3D(pool_size=3, strides=2, padding="same")(x)

		c2 = Conv3D(32, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)
		x = BatchNormalization(axis=1, momentum=0.99)(c2)
		x = keras.layers.LeakyReLU(alpha=0.01)(x)
		x = AveragePooling3D(pool_size=3, strides=2, padding="same")(x)

		c3 = Conv3D(32, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)
		x = BatchNormalization(axis=1, momentum=0.99)(c3)
		x = keras.layers.LeakyReLU(alpha=0.01)(x)
		x = AveragePooling3D(pool_size=3, strides=2, padding="same")(x)
	
		enc = Conv3D(32, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)

		x = BatchNormalization(axis=1, momentum=0.99)(enc)
		res = keras.layers.LeakyReLU(alpha=0.01)(x)

		model = Model(inputs=inputs, outputs=res)
		print(model.summary())
	
		encoded_l = model(left_input)
		encoded_r = model(right_input)

		diff = keras.layers.subtract([encoded_l, encoded_r])
		x = BatchNormalization(axis=1, momentum=0.99)(diff)
		x = keras.layers.LeakyReLU(alpha=0.01)(x)
		x = Flatten()(x)
		dense1 = Dense(2048, kernel_regularizer=keras.regularizers.l2(reg))(x)
		x = BatchNormalization(momentum=0.99)(dense1)
		x = keras.layers.LeakyReLU(alpha=0.01)(x)
		x = Dense(1024, kernel_regularizer=keras.regularizers.l2(reg))(x)
		x = BatchNormalization(momentum=0.99)(x)
		x = keras.layers.LeakyReLU(alpha=0.01)(x)
		x = Dropout(0.5)(x)
		prediction = Dense(2,activation='softmax')(x)
		siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

	print("network ok")
	gc.collect()
	parallel_model = multi_gpu_model(siamese_net,gpus=2) #distribute model over 2 gpus

	adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.000000001, amsgrad=False)
	parallel_model.compile(optimizer=adam,
		          loss='categorical_crossentropy',
		          metrics=['accuracy','mean_squared_error','mean_squared_logarithmic_error',])
	siamese_net.summary()
	gc.collect()

	iter1 = datatrain.make_initializable_iterator() # iterator for training set
	next1 = iter1.get_next()
	iter2 = dataval.make_initializable_iterator() # iterator for validation set
	next2 = iter2.get_next()
	sess.run(iter1.initializer, feed_dict={data_path: train_filename})
	sess.run(iter2.initializer, feed_dict={data_path: train_filename})
	print("init iterateur ok")
	
	batch0 = sess.run(next2) # example images
	imgtest = batch0[1][0,:,:,:]
	imgtest2 = batch0[2][0,:,:,:]
	imgtest = imgtest.reshape((-1, 1, xmax, ymax, zmax))
	imgtest2 = imgtest2.reshape((-1, 1, xmax, ymax, zmax))
  
	train_gen = DataGenerator(datatrain, next1, sess, 200, batch_size, True)
	val_gen = DataGenerator(dataval, next2, sess, 56, batch_size, False)
  
  # training and validation
	history = parallel_model.fit_generator(train_gen, epochs=epochs, verbose=1, validation_data=val_gen, max_queue_size=40, workers=12, use_multiprocessing=False, callbacks=[keras.callbacks.CSVLogger("log.csv", separator=',', append=True)])

	#plot loss, accuracy, and error on training and validation set
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train loss', 'Val loss'], loc='upper left')
	plt.show()

	plt.plot(history.history['acc'])
	plt.plot(history.history['mean_squared_error'])
	plt.plot(history.history['mean_squared_logarithmic_error'])
	plt.plot(history.history['val_acc'])
	plt.plot(history.history['val_mean_squared_error'])
	plt.plot(history.history['val_mean_squared_logarithmic_error'])
	plt.ylabel('Accuracy and Error')
	plt.xlabel('Epoch')
	plt.legend(['Train acc', 'Train err', 'Train log err', 'Val acc', 'Val err', 'Val log err'], loc='upper left')
	plt.show()


	plotExampleImage(np.squeeze(imgtest),"input1")
	plotExampleImage(np.squeeze(imgtest2),"input2")
	plt.show()


	siamese_net.save("siamese.h5") #save model and parameters

	sess.close()
