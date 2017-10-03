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

nb_clstm_filter = 30

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length
    return get_output_length(width), get_output_length(height) 


def nn_base(trainable=False):
    def f(input_tensor):
        input_shape = (None, None, 3)

        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

        bn_axis = 3

        x = img_input
        x = Convolution2D(64, (3, 3), name='conv1', padding='same', trainable = trainable, activation='relu')(x)
        x = Convolution2D(64, (3, 3), name='conv4', padding='same', trainable = trainable, activation='relu')(x)
        x = Convolution2D(64, (3, 3), name='conv5', padding='same', trainable = trainable, activation='relu')(x)
        return x
    return f

def rpn(num_anchors):
    def f(base_layers):
        x = Convolution2D(24, (2, 2), padding='same', activation='relu', name='rpn_conv1')(base_layers)

        x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', padding='same', name='rpn_out_class')(x)
        x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', padding='same', name='rpn_out_regress')(x)

        return [x_class, x_regr, base_layers]
    return f


def time_broadcast(f, x):
    shape = tf.shape(x)
    num_videos, num_frames, w, h, c = [shape[i] for i in range(5)]

    time_flat = tf.reshape(x, [-1, w,h,c])

    y = f(time_flat)

    shape = tf.shape(y)
    _, w, h, c = [shape[i] for i in range(4)]
    y = tf.reshape(y, [num_videos, num_frames, w, h, c])
    return y

def build_shared(video_input):
    with tf.name_scope('shared_layers'):
        base = nn_base(trainable=True)
        shared_layers = time_broadcast(base, video_input)

        num_channels = 64

        shared_layers = clstm(shared_layers,num_channels,nb_clstm_filter,3)
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
        return y_cls, y_reg

