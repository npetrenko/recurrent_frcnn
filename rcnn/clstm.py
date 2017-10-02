import tensorflow as tf

def cnn(x, num_channels, nbfilter, filtersize, name):
    finit = tf.random_normal([filtersize, filtersize, num_channels, nbfilter], stddev=0.05)

    filt = tf.get_variable(name + '_' + 'conv_filter', initializer=finit)
    b = tf.get_variable(name + '_' + 'cov_bias', initializer=tf.random_normal([nbfilter], stddev=0.05))
    x = tf.nn.conv2d(x, filt, [1,1,1,1], 'SAME') + b
    return x #tf.nn.relu(x)

def clstm(x, num_channels, n_hidden, filtersize):
    with tf.variable_scope('clstm'):
        batchsize, height, width = tf.shape(x)[0], tf.shape(x)[2], tf.shape(x)[3]
        for scope in ['0', '1', '2', '3']:
            with tf.variable_scope(scope):
                cnn(x[:,0], num_channels, n_hidden, filtersize, '0')
                cnn(tf.ones([1,30,30,n_hidden]), n_hidden, n_hidden, filtersize, '1')

        def step(prev, x):
            st_1, ct_1 = tf.unstack(prev)

            with tf.variable_scope('0', reuse=True):
                tmp0 = cnn(x, num_channels, n_hidden, filtersize, '0')
                tmp1 = cnn(st_1, n_hidden, n_hidden, filtersize, '1')
            i = tf.sigmoid(tmp0 + tmp1)

            with tf.variable_scope('1', reuse=True):
                tmp0 = cnn(x, num_channels, n_hidden, filtersize, '0')
                tmp1 = cnn(st_1, n_hidden, n_hidden, filtersize, '1')
            f = tf.sigmoid(tmp0 + tmp1)

            with tf.variable_scope('2', reuse=True):
                tmp0 = cnn(x, num_channels, n_hidden, filtersize, '0')
                tmp1 = cnn(st_1, n_hidden, n_hidden, filtersize, '1')
            o = tf.sigmoid(tmp0 + tmp1)

            with tf.variable_scope('3', reuse=True):
                tmp0 = cnn(x, num_channels, n_hidden, filtersize, '0')
                tmp1 = cnn(st_1, n_hidden, n_hidden, filtersize, '1')
            g = tf.sigmoid(tmp0 + tmp1)

            ct = ct_1*f + g*i

            st = tf.tanh(ct)*o
            return tf.stack([st, ct])
        states = tf.scan(step, tf.transpose(x, [1,0,2,3,4]),
                         initializer=tf.random_normal([2,batchsize, height, width, n_hidden], stddev=0.1))
        states = states[:,0]
        states = tf.transpose(states, [1,0,2,3,4])
    return states
