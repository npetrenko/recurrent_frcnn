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

import tensorflow as tf
from rcnn.clstm import clstm
from rcnn import detector_rpn_extraction
from rcnn.generate_cache import create_cache

sess = tf.Session()
K.set_session(sess)

sys.setrecursionlimit(40000)

video_path = './videos'
annotation_path = './annotations'
num_rois = 32
num_epochs = 2000
config_filename = 'config.pickle'
output_weight_path = './save_dir/rpn_only.sv'
n_jobs = 8

from rcnn.video_parser import get_data

C = config.Config()

C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

C.model_path = output_weight_path
C.num_rois = int(num_rois)

from rcnn import simple_nn as nn
C.network = 'simple_nn'

all_videos, classes_count, class_mapping = get_data(video_path, annotation_path)

if not os.path.exists(os.path.join(C.tmp_dir, 'rpn_tmp', '0')):
    print('Cache not found! Creating cache')
    t0 = time.time()
    create_cache(all_videos, classes_count, C, nn.get_img_output_length, n_jobs=n_jobs)
    print('Generating cache took {} minutes'.format((time.time() - t0)/60))

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

num_classes = len(C.class_mapping)

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

with open(config_filename, 'wb') as config_f:
    pickle.dump(C,config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_filename))

#random.shuffle(all_videos)

num_videos = len(all_videos)

print('Num samples {}'.format(num_videos))


data_gen = data_generators.video_streamer(all_videos, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

video_input = tf.placeholder(tf.float32, [None,None,None,None,3], name='video_input')
rpn_target_cls = tf.placeholder(tf.float32, [None,None,None,None,2*num_anchors], name='rpn_target_for_classification')
rpn_target_reg = tf.placeholder(tf.float32, [None,None,None,None,2*num_anchors*4], name='rpn_target_for_regression')

detector_selected_time = tf.placeholder(tf.int32, name='selector_timestep_for_detector')
roi_input = tf.placeholder(tf.int64, [None,None,4], name='roi_input')

y1_input = tf.placeholder(tf.float32, [None,None,num_classes], name='detector_clf_input')
y2_input = tf.placeholder(tf.float32, [None,None,(num_classes-1)*4*2], name='detector_regr_input')

shared = nn.build_shared(video_input)

rpn = nn.build_rpn(shared, num_anchors)


classifier = nn.classifier(roi_input, C.num_rois, nb_classes=num_classes, trainable=True)(shared[:, detector_selected_time])


rpn_clf_loss = losses.rpn_loss_cls(num_anchors)(rpn_target_cls, rpn[0])
rpn_loss = losses.rpn_loss_regr(num_anchors)(rpn_target_reg, rpn[1]) \
        + rpn_clf_loss

rpn_summary = [tf.summary.scalar('rpn_loss', rpn_loss), tf.summary.scalar('rpn_clf_loss', rpn_clf_loss),
            tf.summary.scalar('rpn_reg_loss', rpn_loss - rpn_clf_loss)]
rpn_summary = tf.summary.merge(rpn_summary)

detector_clf_loss = losses.class_loss_cls(y1_input, classifier[0], detector_selected_time)

detector_reg_loss = losses.class_loss_regr(num_classes-1)(y2_input, classifier[1], detector_selected_time)

detector_loss = detector_reg_loss + detector_clf_loss

detector_summary = [tf.summary.scalar('detector_loss',detector_loss),tf.summary.scalar('detector_clf_loss', detector_clf_loss),tf.summary.scalar('detector_reg_loss',detector_reg_loss)]

detector_loss_summary = tf.summary.merge(detector_summary)

writer = tf.summary.FileWriter('/tmp/clstm')

rpn[0] = tf.nn.sigmoid(rpn[0])
classifier[0] = tf.nn.softmax(classifier[0])

def predict_rpn(X):
    return sess.run(rpn, {video_input: X})

def generate_train_op(loss):
    optimizer = tf.train.AdamOptimizer(0.00002)
    gvs = optimizer.compute_gradients(loss)
    capped = [(tf.clip_by_value(grad, -30, 30), var) for grad, var in gvs if grad is not None]
    return optimizer.apply_gradients(capped)

rpn_train_op = generate_train_op(rpn_loss)
detector_train_op = generate_train_op(detector_loss)

def run_rpn(X, Y):
    summary, _ = sess.run([rpn_summary, rpn_train_op], {video_input: X, rpn_target_cls: Y[0], rpn_target_reg: Y[1]}) 
    return summary

def run_detec(X, ROI, Y1, Y2, timestep):
    summary, _ = sess.run([detector_loss_summary, detector_train_op], {video_input:X, roi_input:ROI, y1_input:Y1, y2_input:Y2, detector_selected_time:timestep})
    return summary

epoch_length = 1000
num_epochs = int(num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

init = tf.global_variables_initializer()
sess.run(init)

writer.add_graph(sess.graph)

saver = tf.train.Saver()

try:
    saver.restore(sess, output_weight_path)
except tf.errors.NotFoundError:
    print('Failed to load weights!')

for epoch_num in range(num_epochs):

    iii = 0

    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    while True:
        try:

            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            t0 = time.time()
            X, Y, img_data = next(data_gen)

            writer.add_summary(run_rpn(X,Y))

            P_rpn = predict_rpn(X)

            tlen = P_rpn[0].shape[1]

            timestep = np.random.randint(low=0,high=tlen)

            P_rpn = list(map(lambda x: x[:,timestep], P_rpn))

            ROI, YY = detector_rpn_extraction.extract_features(P_rpn, C, img_data[0][timestep])

            if iii % 4 == 0:
                saver.save(sess, output_weight_path)

            if ROI is None:
                print('ROI is none. Skipping detertor training')
                continue

            Y1, Y2 = YY

            writer.add_summary(run_detec(X, ROI, Y1, Y2, timestep))

            iii += 1
        except Exception as e:
            print('Exception: {}'.format(e))
            raise

print('Training complete, exiting.')
