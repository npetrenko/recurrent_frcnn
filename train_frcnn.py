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

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

sess = tf.Session()
K.set_session(sess)

sys.setrecursionlimit(40000)

video_path = ['/tmp/MOT17']
#video_path = ['/u01/tmp/MOT17_test/', '/u01/tmp/newCam_test/']
#annotation_path = './annotations'
num_rois = 32
num_epochs = 2000
config_filename = 'config.pickle'

pretrained_base = '../vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
output_weight_path = './experiment_save/with_det'#'./save_dir/rpn_only.sv'
n_jobs = 4

tensorboard_dir = '/tmp/clstm'

#from rcnn.video_parser import get_data
from rcnn.MOT_parser import get_data

C = config.Config()

C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

C.model_path = output_weight_path
C.num_rois = int(num_rois)


from rcnn import simple_nn as nn
C.network = 'simple_nn'

# parse video data
all_videos, classes_count, class_mapping = get_data(video_path,part='train')

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

#random.shuffle(all_videos) # maybe we should shuffle them all sometimes?

num_videos = len(all_videos)

print('Num samples {}'.format(num_videos))

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

global_step = tf.Variable(0, name='global_step', trainable=False)

model = nn.FRCNN(num_anchors, C.num_rois, base_weights = pretrained_base, global_step=global_step, lr = 0.00001, kl_ratio=0.01)

# if it fails to find a folder in rpn_tmp it will generate the whole cache again
if not os.path.exists(os.path.join(C.tmp_dir, 'rpn_tmp', '0')):
    print('Cache not found! Creating cache')
    t0 = time.time()
    create_cache([x['video'] for x in all_videos], classes_count, C, model.get_img_output_length, n_jobs=n_jobs)
    print('Generating cache took {} minutes'.format((time.time() - t0)/60))


data_gen = data_generators.video_streamer(all_videos, classes_count, C, model.get_img_output_length, K.image_dim_ordering(), mode='train', frame_batchsize=8)

writer = tf.summary.FileWriter(tensorboard_dir)

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

init = tf.global_variables_initializer()
sess.run(init)
model.init_weights()

writer.add_graph(sess.graph)

saver = tf.train.Saver()

try:
    saver.restore(sess, output_weight_path)
except tf.errors.NotFoundError:
    print('Failed to load weights!')

batch_num = 0

with sess.as_default():
    while True:
        try:
            X, Y, img_data = next(data_gen)

            writer.add_summary(model.train_rpn(X,Y), global_step=sess.run(global_step))

            P_rpn = model.predict_rpn(X)

            tlen = P_rpn[0].shape[1]

            timestep = np.random.randint(low=0,high=tlen)

            P_rpn = list(map(lambda x: x[:,timestep], P_rpn))

            ROI, YY = detector_rpn_extraction.extract_features(P_rpn, C, img_data[0][timestep])

            if batch_num % 200 == 0:
                saver.save(sess, output_weight_path)

            if ROI is None:
                print('ROI is none. Skipping detertor training')
                continue

            Y1, Y2 = YY

            writer.add_summary(model.train_detec(X, ROI, Y1, Y2, timestep), global_step=sess.run(global_step))

            batch_num += 1
        except Exception as e:
            print('Exception: {}'.format(e))
            raise
