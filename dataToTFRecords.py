"""
<Script to read Nifty data and corresponding labels in CSV file, and save data as tensorflow TFRecords format>
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
from deepbrain import Extractor

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

def loadNifti(imgname, ext, f):
	img = nib.load(imgname)
	data = img.get_fdata()
	data = cropAroundBrain(data, ext)
	data = np.asarray(data).astype(np.float32).reshape((data.shape[0], data.shape[1], data.shape[2]))
	f.write(str(data.shape[0])+"\t"+str(data.shape[1])+"\t"+str(data.shape[2])+"\n")
	zmax = 150
	xmax = 205
	ymax = 216
	if data.shape[0] < xmax: #zero padding one side if z < zmax
		data=np.pad(data, ((((xmax-data.shape[0])//2)+((xmax-data.shape[0])%2),((xmax-data.shape[0])//2)), (0,0), (0,0)), 'minimum')
	if data.shape[1] < ymax: #zero padding one side if z < zmax
		data=np.pad(data, ((0,0), (((ymax-data.shape[1])//2)+((ymax-data.shape[1])%2),((ymax-data.shape[1])//2)), (0,0)), 'minimum')
	if data.shape[2] < zmax: #zero padding one side if z < zmax
		data=np.pad(data, ((0,0), (0,0), (((zmax-data.shape[2])//2)+((zmax-data.shape[2])%2),((zmax-data.shape[2])//2))), 'minimum')
	#data = zoom(data,(0.5,0.5,0.5))
	#mask = range(0,zmax,2)
	#data = np.delete(data,mask, axis=2)
	data = (255. / 4095.) * data
	data = data / 255.
	data = zoom(data,(0.5,0.5,0.5))
	return data

def cropAroundBrain(image, ext):
	prob = ext.run(image) 
	mask = prob > 0.6
	np.putmask(image, prob < 0.6, np.min(image))
	mask = mask.astype(np.uint8)
	D00, H00, D01, H01 = rectangleAroundBrainInAxis(mask, 0)
	W00, D10, W01, D11 = rectangleAroundBrainInAxis(mask, 1)
	W10, H10, W11, H11 = rectangleAroundBrainInAxis(mask, 2)
	W0 = min(D00, W00)
	W1 = max(D01, W01)
	D0 = min(W10, H00)
	D1 = max(W11, H01)
	H0 = min(H10, D10)
	H1 = max(H11, D11)
	o = 10
	croped = image[H0-o: H0+H1+o, D0-o: D0+D1+o, W0-o: W0+W1+o]
	return croped

def rectangleAroundBrainInAxis(mask, axis):
	proj = np.max(mask, axis=axis)
	contours, hierarchy = cv2.findContours(proj,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
	x,y,w,h = cv2.boundingRect(cont_sorted[0])
	
	return x,y,w,h

ext = Extractor()
traj = {}
f = open("/home/cecilia/LSN/legacy_code/ADNI_trajectory_labels_4class_MMSE_3cstp_from_m72_autoselect.csv")
for line in f.readlines():
	fields=line.split(",")
	try:
		traj[fields[1]]=int(fields[17][0])
	except ValueError:
		continue
f.close()
shuffle_data = True  # shuffle the addresses before saving
path = '/home/cecilia/ADNI_2Yr_/*Screening*.nii'
# read addresses and labels from the 'train' folder
addrs = glob.glob(path)
labels = []
for addr in addrs :
	try:
		labels.append(traj[addr[len("/home/cecilia/ADNI_2Yr_/"):len("/home/cecilia/ADNI_2Yr_/")+10]])
	except KeyError:
		print("Error")
		continue
	
print("Nb of Stable subjects: ",labels.count(0))
print("Nb of Decline subjects: ",labels.count(1))
# to shuffle data
if shuffle_data:
	c = list(zip(addrs, labels))
	shuffle(c)
	addrs, labels = zip(*c)

train_addrs = addrs
train_labels = labels

train_filename = 'train_siamese_resize.tfrecords'  # address to save the TFRecords file

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

f=open("cropedBrains.csv","w")
for i in range(len(train_addrs)):
	print('Train data: {}/{}'.format(i+1, len(train_addrs)))
	sys.stdout.flush()

	# Load the image
	try :
		img = loadNifti(train_addrs[i], ext, f)
		img2 = loadNifti(train_addrs[i].replace("Screening","Month 12"), ext, f)
	except Exception:
		print("Error")
		print(train_addrs[i])
		continue

	label = train_labels[i]

	# Create a feature
	feature = {'train/label': _int64_feature(label),
	       'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
	       'train/image2': _bytes_feature(tf.compat.as_bytes(img2.tostring()))}

	# Create an example protocol buffer
	example = tf.train.Example(features=tf.train.Features(feature=feature))

	# Serialize to string and write on the file
	writer.write(example.SerializeToString())
f.close()    
writer.close()
sys.stdout.flush()
print("tfrecord file complete")
