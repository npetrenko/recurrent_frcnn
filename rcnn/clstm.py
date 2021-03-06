import tensorflow as tf

def cnn(x, nbfilter, filtersize, name, use_bias=True, num_channels=None):
    if num_channels is None:
        num_channels = int(x.shape[-1])

    #print([filtersize, filtersize, num_channels, nbfilter])

    finit = tf.random_normal([filtersize, filtersize, num_channels, nbfilter], stddev=0.05)

    filt = tf.get_variable(name + '_' + 'conv_filter', initializer=finit)
    x = tf.nn.conv2d(x, filt, [1,1,1,1], 'SAME')

    if use_bias:
        b = tf.get_variable(name + '_' + 'cov_bias', initializer=tf.random_normal([nbfilter], stddev=0.05))
        x += b

    x.set_shape([None,None,None,nbfilter])
    return x 

def clstm(x, n_hidden, filtersize, name):
    num_channels = int(x.shape[-1])

    with tf.variable_scope('clstm_'+name):
        batchsize, height, width = tf.shape(x)[0], tf.shape(x)[2], tf.shape(x)[3]
        for scope in ['0', '1', '2', '3']:
            with tf.variable_scope(scope):
                cnn(x[:,0], n_hidden, filtersize, '0')
                cnn(tf.ones([1,30,30,n_hidden]), n_hidden, filtersize, '1', use_bias=False)

        def step(prev, x):
            st_1, ct_1 = tf.unstack(prev)

            with tf.variable_scope('0', reuse=True):
                tmp0 = cnn(x, n_hidden, filtersize, '0')
                tmp1 = cnn(st_1, n_hidden, filtersize, '1', use_bias=False)
            i = tf.sigmoid(tmp0 + tmp1)

            with tf.variable_scope('1', reuse=True):
                tmp0 = cnn(x, n_hidden, filtersize, '0')
                tmp1 = cnn(st_1, n_hidden, filtersize, '1', use_bias=False)
            f = tf.sigmoid(tmp0 + tmp1)

            with tf.variable_scope('2', reuse=True):
                tmp0 = cnn(x, n_hidden, filtersize, '0')
                tmp1 = cnn(st_1, n_hidden, filtersize, '1', use_bias=False)
            o = tf.sigmoid(tmp0 + tmp1)

            with tf.variable_scope('3', reuse=True):
                tmp0 = cnn(x, n_hidden, filtersize, '0')
                tmp1 = cnn(st_1, n_hidden, filtersize, '1', use_bias=False)

            g = tf.tanh(tmp0 + tmp1)

            ct = ct_1*f + g*i

            st = tf.tanh(ct)*o
            return tf.stack([st, ct])

        states = tf.scan(step, tf.transpose(x, [1,0,2,3,4]),
                         initializer=tf.random_normal([2,batchsize, height, width, n_hidden], stddev=0.1))
        states = states[:,0]
        states = tf.transpose(states, [1,0,2,3,4])
        states.set_shape([None,None,None,None,n_hidden])
    return states
