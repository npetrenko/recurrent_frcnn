from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from rcnn import config, data_generators
from rcnn import losses as losses
import rcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras.layers import TimeDistributed, Lambda

import tensorflow as tf
from rcnn.clstm import clstm
from rcnn import detector_rpn_extraction

sess = tf.Session()
K.set_session(sess)

sys.setrecursionlimit(40000)

parser = OptionParser()

video_path = './videos'
annotation_path = './annotations'
num_rois = 32
num_epochs = 2000
config_filename = 'config.pickle'
output_weight_path = './save_dir/rpn_only.sv'
input_weight_path = None

from rcnn.video_parser import get_data

C = config.Config()

C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

C.model_path = output_weight_path
C.num_rois = int(num_rois)

from rcnn import simple_nn as nn
C.network = 'simple_nn'

# check if weight path was passed via command line
if input_weight_path:
    C.base_net_weights = input_weight_path

all_videos, classes_count, class_mapping = get_data(video_path, annotation_path)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

num_classes = len(C.class_mapping)

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C,config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_videos)

num_imgs = len(all_videos)

#train_videos = [s for s in all_videos if s['imageset'] == 'trainval']
#val_videos = [s for s in all_videos if s['imageset'] == 'test']
train_videos = all_videos
val_videos = all_videos

print('Num train samples {}'.format(len(train_videos)))
print('Num val samples {}'.format(len(val_videos)))


data_gen_train = data_generators.video_streamer(train_videos, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.video_streamer(val_videos, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')

input_shape_img = (None, None, None, 3)

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

video_input = tf.placeholder(tf.float32, [None,None,None,None,3])
rpn_target_cls = tf.placeholder(tf.float32, [None,None,None,None,2*num_anchors])
rpn_target_reg = tf.placeholder(tf.float32, [None,None,None,None,2*num_anchors*4])

detector_selected_time = tf.placeholder(tf.int32)
roi_input = tf.placeholder(tf.int64, [None,None,4])

y1_input = tf.placeholder(tf.float32, [None,None,num_classes])
y2_input = tf.placeholder(tf.float32, [None,None,(num_classes-1)*4*2])

shared = nn.build_shared(video_input)

rpn = nn.build_rpn(shared, num_anchors)

def predict_rpn(X):
    return sess.run(rpn, {video_input: X})

print('\n\n\nShared shape: {}'.format(num_classes))

classifier = nn.classifier(roi_input, C.num_rois, nb_classes=num_classes, trainable=True)(shared[:, detector_selected_time])

#model_rpn = Model(img_input, rpn[:2])
#model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
#model_all = Model([img_input, roi_input], rpn[:2] + classifier)


rpn_loss = losses.rpn_loss_regr(num_anchors)(rpn_target_reg, rpn[1]) \
        + losses.rpn_loss_cls(num_anchors)(rpn_target_cls, rpn[0])

rpn_loss_summary = tf.summary.scalar('rpn_loss', rpn_loss)

print(classifier[0].shape)
print(classifier[1].shape)

detector_loss = losses.class_loss_cls(y1_input, classifier[0], detector_selected_time) + \
            losses.class_loss_regr(num_classes-1)(y2_input, classifier[1], detector_selected_time)

detector_loss_summary = tf.summary.scalar('detector_loss', detector_loss)

#all_summ = tf.summary.merge_all()

writer = tf.summary.FileWriter('/tmp/clstm')

def generate_train_op(loss):
    optimizer = tf.train.AdamOptimizer(0.0002)
    return optimizer.minimize(loss)

rpn_train_op = generate_train_op(rpn_loss)
detector_train_op = generate_train_op(detector_loss)

def run_rpn(X, Y):
    summary, _ = sess.run([rpn_loss_summary, rpn_train_op], {video_input: X, rpn_target_cls: Y[0], rpn_target_reg: Y[1]}) 
    return summary

def run_detec(X, ROI, Y1, Y2, timestep):
    summary, _ = sess.run([detector_loss_summary, detector_train_op], {video_input:X, roi_input:ROI, y1_input:Y1, y2_input:Y2, detector_selected_time:timestep})
    return summary

#model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])

#model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
#model_all.compile(optimizer='sgd', loss='mae')

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

saver = tf.train.Saver()

try:
    saver.restore(sess, output_weight_path)
except tf.errors.NotFoundError:
    print('Failed to load weights!')

for epoch_num in range(num_epochs):

    iii = 0

    progbar = generic_utils.Progbar(epoch_length)
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
            X, Y, img_data = next(data_gen_train)

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
            #P_rpn = predict_rpn(X)


            #losses[iter_num, 0] = loss_rpn[1]
            #losses[iter_num, 1] = loss_rpn[2]

            #if use_detector:
                #losses[iter_num, 2] = loss_class[1]
                #losses[iter_num, 3] = loss_class[2]
                #losses[iter_num, 4] = loss_class[3]

            #iter_num += 1

            #if use_detector:
                #progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])), ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])
            #else:
                #progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1]))])

            #if iter_num == epoch_length:
                #loss_rpn_cls = np.mean(losses[:, 0])
                #loss_rpn_regr = np.mean(losses[:, 1])
                #if use_detector:
                    #loss_class_cls = np.mean(losses[:, 2])
                    #loss_class_regr = np.mean(losses[:, 3])
                    #class_acc = np.mean(losses[:, 4])

                #mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                #rpn_accuracy_for_epoch = []

                #if C.verbose:
                    #print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    #print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    #print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    #print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    #if use_detector:
                        #print('Loss Detector classifier: {}'.format(loss_class_cls))
                        #print('Loss Detector regression: {}'.format(loss_class_regr))
                    #print('Elapsed time: {}'.format(time.time() - start_time))

                    #if not use_detector:
                        #loss_class_cls = 0
                        #loss_class_regr = 0
                #curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                #iter_num = 0
                #start_time = time.time()

                #if curr_loss < best_loss:
                    #if C.verbose:
                        #print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                    #best_loss = curr_loss
                    #model_all.save_weights(C.model_path)

                #break

        except Exception as e:
            print('Exception: {}'.format(e))
            raise

print('Training complete, exiting.')
