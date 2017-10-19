# -*- coding: utf-8 -*-
''' Simple nn for testing'''

from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed

from keras import backend as K

from .clstm import clstm

from rcnn.RoiPoolingConv import RoiPoolingConv
from rcnn.FixedBatchNormalization import FixedBatchNormalization

nb_clstm_filter = 64
shared_dim = nb_clstm_filter

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//4
    return get_output_length(width), get_output_length(height) 


def nn_base(trainable=False):
    def f(input_tensor):

        bn_axis = 3

        x = input_tensor
        x = Convolution2D(32, (4, 4), name='conv1', padding='same', trainable = trainable, activation='relu')(x)
        x = Convolution2D(32, (4, 4), name='conv2', padding='same', trainable = trainable, activation='relu')(x)
        x = MaxPooling2D((2,2))(x)
        x = Convolution2D(64, (4, 4), name='conv3', padding='same', trainable = trainable, activation='relu')(x)
        x = Convolution2D(128, (4, 4), name='conv4', padding='same', trainable = trainable, activation='relu')(x)
        x = MaxPooling2D((2,2))(x)
        x = Convolution2D(128, (4, 4), name='conv5', padding='same', trainable = trainable, activation='relu')(x)
        return x
    return f

def rpn(num_anchors):
    def f(base_layers):
        x = Convolution2D(32, (2, 2), padding='same', activation='relu', name='rpn_conv1')(base_layers)

        x_class = Convolution2D(num_anchors, (1, 1), activation='linear', padding='same', name='rpn_out_class')(x)
        x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', padding='same', name='rpn_out_regress')(x)

        return [x_class, x_regr, base_layers]
    return f


def time_broadcast(f, x):
    shape = tf.shape(x)
    num_videos, num_frames, w, h, c = [shape[i] for i in range(5)]

    time_flat = tf.reshape(x, [-1, w,h,c])

    time_flat.set_shape([None,None,None,x.shape[-1]])

    y = f(time_flat)

    shape = tf.shape(y)
    _, w, h, c = [shape[i] for i in range(4)]
    y = tf.reshape(y, [num_videos, num_frames, w, h, c])
    return y

def build_shared(video_input):
    with tf.name_scope('shared_layers'):
        base = nn_base(trainable=True)

        shared_layers = time_broadcast(base, video_input)

        num_channels = 128

        shared_layers = clstm(shared_layers,num_channels,nb_clstm_filter,6, 'forward_clstm')
        shared_layers = clstm(shared_layers[:,::-1],nb_clstm_filter,nb_clstm_filter,6, 'backward_cltsm')[:,::-1]
    return shared_layers

def build_rpn(x, num_anchors):
    with tf.name_scope('RPN'):
        
        shape = tf.shape(x)
        num_videos, num_frames, w, h, c = [shape[i] for i in range(5)]
        c = nb_clstm_filter

        time_flat = tf.reshape(x, [-1, w,h,c])

        y_cls, y_reg, _ = rpn(num_anchors)(time_flat)

        shape = tf.shape(y_cls)
        _, w, h, c = [shape[i] for i in range(4)]

        y_cls = tf.reshape(y_cls, [num_videos, num_frames, w, h, c])
        y_reg = tf.reshape(y_reg, [num_videos, num_frames, w, h, c*4])
        return [y_cls, y_reg]

def classifier_layers(x, input_shape, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    # (hence a smaller stride in the region that follows the ROI pool)
    x = TimeDistributed(Convolution2D(30, 3, padding='same', activation='relu', trainable=trainable, input_shape=input_shape))(x)
    x = TimeDistributed(Convolution2D(30, 3, padding='same', activation='relu', trainable=trainable))(x)

    return x

def classifier(input_rois, num_rois, nb_classes, trainable=False):
    def f(base_layers):

        pooling_regions = 14
        input_shape = (num_rois,14,14,64)

        out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
        out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

        out = TimeDistributed(Flatten())(out)

        out_class = TimeDistributed(Dense(nb_classes, activation='linear', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
        return [out_class, out_regr]
    return f
