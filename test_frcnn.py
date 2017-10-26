from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
import pickle
import os

from keras import backend as K
from keras.optimizers import Adam
from rcnn import config, data_generators
from rcnn import losses as losses
import rcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from rcnn.MOT_parser import read_ini
import cv2

import tensorflow as tf
from rcnn import detector_rpn_extraction
from rcnn.generate_cache import create_cache

from rcnn import config
from rcnn import roi_helpers

from os.path import join
from os import listdir

os.environ['CUDA_VISIBLE_DEVICES'] = ""

sess = tf.Session()
K.set_session(sess)

sys.setrecursionlimit(40000)

dataset_path = ['/tmp/MOT17_test/']
part = 'train'
num_rois = 32
config_filename = 'config.pickle'

save_path = './experiment_save/with_det'#'./save_dir/rpn_only.sv'
n_jobs = 4

seqs = [join(dataset, part, video) for dataset in dataset_path for video in listdir(join(dataset, part)) if 'DS_' not in video and 'DS_' not in dataset]
print(seqs)

sess = tf.Session()
K.set_session(sess)

with open(config_filename, 'rb') as f_in:
	C = pickle.load(f_in)

from rcnn import simple_nn as nn

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)


# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

class_mapping = C.class_mapping

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)

class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = num_rois

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

model = nn.FRCNN(num_anchors, C.num_rois, base_weights=None, lr = 0.00001)

print('Loading weights from {}'.format(save_path))
saver = tf.train.Saver()
saver.restore(sess, save_path)

all_imgs = []

classes = {}

bbox_threshold = 0.76

with sess.as_default():
    for video_seq in seqs:
        seqinfo = read_ini(join(video_seq, 'seqinfo.ini'))

        img_path = join(video_seq, seqinfo['imDir'])

        try:
            coord_form = seqinfo['coordform']
        except KeyError:
            coord_form = None

        num_gt_objects = 0
        with open(join(video_seq, 'gt/gt.txt'), 'r') as f:
            for line in f:
                #frameix,x1,y1,w,h = map(lambda x: int(float(x)), line_split[0:1] + line_split[2:6])
                num_gt_objects += 1

        im_seq = []
        bbox_seq = []

        st = time.time()

        for idx, img_name in enumerate(sorted(os.listdir(img_path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(img_name)

            filepath = join(img_path,img_name)

            img = cv2.imread(filepath)

            X, ratio = format_img(img, C)

            X = np.transpose(X, (0, 2, 3, 1))
            im_seq.append(X[0])

        im_seq = np.array(im_seq)[np.newaxis,...]

        # get the feature maps and output from the RPN
        Y1_seq, Y2_seq, F_seq = model.predict_rpn_base(im_seq)


        for t in range(Y1_seq.shape[1]):
            bbox_seq.append([])
            Y1, Y2, F = Y1_seq[:,t], Y2_seq[:,t], F_seq[:,t]
            R = roi_helpers.rpn_to_roi(Y1, Y2, C, 'tf', overlap_thresh=0.7)

            # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0]//C.num_rois + 1):
                ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0]//C.num_rois:
                    #pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = model.predict_detec(F, ROIs)

                for ii in range(P_cls.shape[1]):

                    #if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        #continue
                    if P_cls[0,ii,0] < bbox_threshold:
                        continue

                    cls_name = class_mapping[0]
                    #cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    #cls_num = np.argmax(P_cls[0, ii, :])
                    cls_num = 0

                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                        tx /= C.classifier_regr_std[0]
                        ty /= C.classifier_regr_std[1]
                        tw /= C.classifier_regr_std[2]
                        th /= C.classifier_regr_std[3]
                        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []

            for key in bboxes:
                bbox = np.array(bboxes[key])
                bbox_seq[-1].append(bbox)

                new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk,:]

                    (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                    #cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

                    textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                    all_dets.append((key,100*new_probs[jk]))

                    (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                    textOrg = (real_x1, real_y1-0)

                    #cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                    #cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                    #cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        print('Elapsed time = {}'.format(time.time() - st))
        print('GT/DET ratio = {}'.format(sum([len(x) for x in bbox_seq])/num_gt_objects))
        #cv2.imshow('img', img)
        #cv2.waitKey(0)
            # cv2.imwrite('./results_imgs/{}.png'.format(idx),img)
