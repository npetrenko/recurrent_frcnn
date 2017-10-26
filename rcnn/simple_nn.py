# -*- coding: utf-8 -*-
''' Simple nn for testing'''

from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from keras.layers import Input, Conv2D, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed

from keras import backend as K
from keras.models import Model

from .clstm import clstm, cnn
from rcnn import losses

from rcnn.RoiPoolingConv import RoiPoolingConv
from rcnn.FixedBatchNormalization import FixedBatchNormalization
import tensorflow as tf

from functools import reduce

nb_clstm_filter = 64

def generate_train_op(loss, lr, global_step=None):
    optimizer = tf.train.AdamOptimizer(lr)
    gvs = optimizer.compute_gradients(loss)
    capped = [(tf.clip_by_value(grad, -30, 30), var) for grad, var in gvs if grad is not None]

    if global_step is None:
        return optimizer.apply_gradients(capped)

    return optimizer.apply_gradients(capped, global_step=global_step)

class FRCNN:
    def __init__(self, num_anchors, num_rois, kl_ratio, base_weights=None, learn_base=False, global_step=None, lr=None):
        self.num_anchors = num_anchors
        self.base_weights = base_weights

        num_classes = 2

        with tf.name_scope('input_placeholders'):
            self.video_input = tf.placeholder(tf.float32, [None,None,None,None,3], name='video_input')
            self.rpn_target_cls = tf.placeholder(tf.float32, [None,None,None,None,2*num_anchors], name='rpn_target_for_classification')
            self.rpn_target_reg = tf.placeholder(tf.float32, [None,None,None,None,2*num_anchors*4], name='rpn_target_for_regression')

            self.detector_selected_time = tf.placeholder(tf.int32, name='selector_timestep_for_detector')
            self.roi_input = tf.placeholder(tf.int64, [None,None,4], name='roi_input')

            self.detector_clf_target = tf.placeholder(tf.float32, [None,None,num_classes], name='detector_clf_target')
            self.detector_regr_target = tf.placeholder(tf.float32, [None,None,(num_classes-1)*4*2], name='detector_regr_target')

        base_layers = tf.identity(self.build_shared(self.video_input, stop_gradient=not learn_base), name='base_layers_output')

        rpn_cls, rpn_reg = self.build_rpn(base_layers, num_anchors)

        base_layers_st = tf.placeholder_with_default(base_layers[:, self.detector_selected_time], shape=[None,None,None,int(base_layers.shape[-1])], name='base_layers_placeholder')

        detector_cls, detector_reg = self.classifier(self.roi_input, num_rois, nb_classes=2, trainable=True)(base_layers_st)
        
        with tf.name_scope('rpn_loss'):
            graph = tf.get_default_graph()
            kls = graph.get_collection('kls')
            rpn_kl_base = sum([x for x in kls if 'BASE' in x.name])/2/kl_ratio
            rpn_kl_rpn = sum([x for x in kls if 'RPN' in x.name])/kl_ratio

            rpn_clf_loss = losses.rpn_loss_cls(num_anchors)(self.rpn_target_cls, rpn_cls)
            rpn_reg_loss = losses.rpn_loss_regr(num_anchors)(self.rpn_target_reg, rpn_reg)
            
            rpn_loss = rpn_clf_loss + rpn_reg_loss + rpn_kl_base + rpn_kl_rpn

        rpn_summary = [tf.summary.scalar('rpn_loss', rpn_loss),
                       tf.summary.scalar('rpn_clf_loss', rpn_clf_loss),
                       tf.summary.scalar('rpn_reg_loss', rpn_reg_loss)]

        self.rpn_summary = tf.summary.merge(rpn_summary)

        with tf.name_scope('detector_loss'):
            graph = tf.get_default_graph()
            kls = graph.get_collection('kls')
            detec_kl_base = sum([x for x in kls if 'BASE' in x.name])/2/kl_ratio
            detec_kl_detec = sum([x for x in kls if 'DETECTOR' in x.name])/kl_ratio

            detector_clf_loss = losses.class_loss_cls(self.detector_clf_target, detector_cls)
            detector_reg_loss = losses.class_loss_regr(num_classes-1)(self.detector_regr_target, detector_reg)

            detector_loss = detector_reg_loss + detector_clf_loss + detec_kl_base + detec_kl_detec

        detector_summary = [tf.summary.scalar('detector_loss',detector_loss),
                            tf.summary.scalar('detector_clf_loss', detector_clf_loss),
                            tf.summary.scalar('detector_reg_loss',detector_reg_loss)]

        self.detector_summary = tf.summary.merge(detector_summary)

        self.rpn_train_op = generate_train_op(rpn_loss, lr=lr, global_step=global_step)
        self.detector_train_op = generate_train_op(detector_loss, lr=lr)

        rpn_cls = tf.identity(tf.nn.sigmoid(rpn_cls), name='rpn_cls_output')
        rpn_reg = tf.identity(rpn_reg, name='rpn_reg_output')

        detector_cls = tf.identity(tf.nn.softmax(detector_cls), name='detector_cls_output')
        detector_reg = tf.identity(detector_reg, name='detector_reg_output')

        self.rpn = [rpn_cls, rpn_reg]

    def init_weights(self):
        self.base_model.load_weights(self.base_weights)

    def train_rpn(self, X, Y):
        sess = tf.get_default_session()
        summary, _ = sess.run([self.rpn_summary, self.rpn_train_op], {self.video_input: X, self.rpn_target_cls: Y[0], self.rpn_target_reg: Y[1]})
        return summary

    def train_detec(self, X, ROI, Y1, Y2, timestep):
        sess = tf.get_default_session()
        summary, _ = sess.run([self.detector_summary, self.detector_train_op], {self.video_input:X, self.roi_input:ROI, 
                                self.detector_clf_target:Y1, self.detector_regr_target:Y2, self.detector_selected_time:timestep})
        return summary

    def predict_rpn(self, X):
        sess = tf.get_default_session()
        return sess.run(self.rpn, {self.video_input: X})

    @staticmethod
    def get_img_output_length(width, height):
            def get_output_length(input_length):
                return input_length//8
            return get_output_length(width), get_output_length(height) 

    @staticmethod
    def nn_base(stop_gradient=False):
        def f(self, input_tensor):
            input_shape = (None, None, 3)
            img_input = Input(tensor=input_tensor, shape=input_shape)

            # Block 1
            x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
            x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

            # Block 2
            x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
            x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

            # Block 3
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
            r = x

            # Block 4
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

            # Block 5
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
            # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

            model = Model(img_input, x)

            self.base_model = model

            if stop_gradient:
                r = tf.stop_gradient(r)

            x = r
            x = cnn(x, 64, 1, name='BASE_c1', activation=tf.nn.relu, bayesian=True)
            x = cnn(x, 64, 3, name='BASE_c2', activation=tf.nn.relu, bayesian=True)
            x = cnn(x, 64, 3, name='BASE_c3', activation=tf.nn.relu, bayesian=True)

            return x
        return f

    def rpn(self, num_anchors):
        def f(base_layers):
            x = cnn(base_layers, 32, 2, name='c0', activation=tf.nn.relu, bayesian=True)

            x_class = cnn(x, num_anchors, 1, name='conv_clf', bayesian=True)
            x_regr = cnn(x, num_anchors*4, 1, name='conv_regr', bayesian=True)

            return [x_class, x_regr, base_layers]
        return f


    def time_broadcast(self, f, x):
        shape = tf.shape(x)
        num_videos, num_frames, w, h, c = [shape[i] for i in range(5)]

        time_flat = tf.reshape(x, [-1, w,h,c])

        time_flat.set_shape([None,None,None,x.shape[-1]])

        y = f(self, time_flat)

        nb_filters = y.shape[-1]

        shape = tf.shape(y)
        _, w, h, c = [shape[i] for i in range(4)]
        y = tf.reshape(y, [num_videos, num_frames, w, h, c])

        y.set_shape([None,None,None,None,int(nb_filters)])
        return y

    def build_shared(self, video_input, stop_gradient):
        with tf.name_scope('shared_layers'):
            base = self.nn_base(stop_gradient=stop_gradient)

            shared_layers = self.time_broadcast(base, video_input)

            shared_layers = clstm(shared_layers,nb_clstm_filter,3, 'forward_clstm', bayesian=True)
            shared_layers = clstm(shared_layers[:,::-1],nb_clstm_filter,3, 'backward_cltsm', bayesian=True)[:,::-1]

        return shared_layers

    def build_rpn(self, base_layers, num_anchors):
        with tf.name_scope('RPN'):
            
            shape = tf.shape(base_layers)
            num_videos, num_frames, w, h, c = [shape[i] for i in range(5)]

            c = int(base_layers.shape[-1])

            time_flat = tf.reshape(base_layers, [-1, w,h,c])

            y_cls, y_reg, _ = self.rpn(num_anchors)(time_flat)

            shape = tf.shape(y_cls)
            _, w, h, c = [shape[i] for i in range(4)]

            y_cls = tf.reshape(y_cls, [num_videos, num_frames, w, h, c])
            y_reg = tf.reshape(y_reg, [num_videos, num_frames, w, h, c*4])
            return [y_cls, y_reg]

    @staticmethod
    def classifier_layers(x, input_shape, trainable=False):
        x = TimeDistributed(Convolution2D(30, 3, padding='same', activation='relu', trainable=trainable, input_shape=input_shape))(x)
        x = TimeDistributed(Convolution2D(30, 3, padding='same', activation='relu', trainable=trainable))(x)

        return x

    def classifier(self, input_rois, num_rois, nb_classes, trainable=False):
        def f(base_layers):
            with tf.name_scope('DETECTOR'):
                pooling_regions = 14
                input_shape = (num_rois,14,14,64)

                out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
                out = self.classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

                out = TimeDistributed(Flatten())(out)

                out_class = TimeDistributed(Dense(nb_classes, activation='linear', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
                # note: no regression target for bg class
                out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
                return [out_class, out_regr]
        return f
