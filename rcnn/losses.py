from keras import backend as K
from keras.objectives import categorical_crossentropy

if K.image_dim_ordering() == 'tf':
    import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 10.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, :, :, 4 * num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        tmp =  lambda_rpn_regr * y_true[:, :, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)) / K.sum(epsilon + y_true[:, :, :, :, :4 * num_anchors])

        return tf.reduce_sum(tmp)

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        tmp = lambda_rpn_class * tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred[:, :, :, :, :], labels=y_true[:, :, :, :, num_anchors:]) 

        ones = y_true[:,:,:,:,num_anchors:]

        to_learn = y_true[:,:,:,:,:num_anchors]

        weight_t = tf.reduce_mean(ones*to_learn) + 0.01
        weight_bg = tf.reduce_mean((1-ones)*to_learn)

        tmp = tf.reduce_sum(tmp*ones*to_learn)/weight_t + tf.reduce_sum(tmp*(1-ones)*to_learn)/weight_bg
        tmp /= (1/weight_t + 1/weight_bg)
        tmp /= (tf.reduce_sum(to_learn) + 0.1)

        return tmp

    return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
    return class_loss_regr_fixed_num

def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true[0, :, :], logits=y_pred[0, :, :]))
