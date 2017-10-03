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
#roi_input = Input(shape=(None, None, 4))

shared = nn.build_shared(video_input)

rpn = nn.build_rpn(shared, num_anchors)

def predict_rpn(X):
    return sess.run(rpn, {video_input: X})
#classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

#model_rpn = Model(img_input, rpn[:2])
#model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
#model_all = Model([img_input, roi_input], rpn[:2] + classifier)


optimizer = tf.train.AdamOptimizer(0.001)

rpn_loss = losses.rpn_loss_regr(num_anchors)(rpn_target_reg, rpn[1]) \
        + losses.rpn_loss_cls(num_anchors)(rpn_target_cls, rpn[0])

tf.summary.scalar('rpn_loss', rpn_loss)

all_summ = tf.summary.merge_all()

writer = tf.summary.FileWriter('/tmp/clstm')

rpn_train_op = optimizer.minimize(rpn_loss)

def run_rpn(X, Y):
    summary, _ = sess.run([all_summ, rpn_train_op], {video_input: X, rpn_target_cls: Y[0], rpn_target_reg: Y[1]}) 
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
            #print('Generating data took {} sec'.format(time.time()-t0))


            #loss_rpn = model_rpn.train_on_batch(X, Y)
            writer.add_summary(run_rpn(X,Y))

            if iii % 40 == 0:
                saver.save(sess, output_weight_path)

            iii += 1
            #P_rpn = predict_rpn(X)

            #R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            #X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

            #if X2 is None:
                #rpn_accuracy_rpn_monitor.append(0)
                #rpn_accuracy_for_epoch.append(0)
                #continue
#
            #neg_samples = np.where(Y1[0, :, -1] == 1)
            #pos_samples = np.where(Y1[0, :, -1] == 0)
            #if len(neg_samples) > 0:
                #neg_samples = neg_samples[0]
            #else:
                #neg_samples = []

            #if len(pos_samples) > 0:
                #pos_samples = pos_samples[0]
            #else:
                #pos_samples = []
            
            #rpn_accuracy_rpn_monitor.append(len(pos_samples))
            #rpn_accuracy_for_epoch.append((len(pos_samples)))

            #use_detector = False
            #if use_detector: #for first runs, do not use detection model
                #if C.num_rois > 1:
                        #if len(pos_samples) < C.num_rois//2:
                                #selected_pos_samples = pos_samples.tolist()
                        #else:
                                #selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                        #try:
                                #selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                        #except:
                                #selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

                        #sel_samples = selected_pos_samples + selected_neg_samples
                #else:
                        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                        #selected_pos_samples = pos_samples.tolist()
                        #selected_neg_samples = neg_samples.tolist()
                        #if np.random.randint(0, 2):
                                #sel_samples = random.choice(neg_samples)
                        #else:
                                #sel_samples = random.choice(pos_samples)

                #loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            #losses[iter_num, 0] = loss_rpn[1]
            #losses[iter_num, 1] = loss_rpn[2]

            #if use_detector:
                #losses[iter_num, 2] = loss_class[1]
                #losses[iter_num, 3] = loss_class[2]
                #losses[iter_num, 4] = loss_class[3]
#
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
