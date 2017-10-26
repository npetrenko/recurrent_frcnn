import tensorflow as tf

def cnn(x, nbfilter, filtersize, name, use_bias=True, num_channels=None, bayesian=False, prior_std=2., activation=None):
    if num_channels is None:
        num_channels = int(x.shape[-1])

    #print([filtersize, filtersize, num_channels, nbfilter])

    filter_shape = [filtersize, filtersize, num_channels, nbfilter]

    finit = tf.random_normal(filter_shape, stddev=0.05)

    filt = tf.get_variable(name + '_' + 'conv_filter_mu', initializer=finit)

    if bayesian:
        graph = tf.get_default_graph()

        filt_logstd = tf.get_variable(name + '_' + 'conv_filter_logsd', initializer=tf.ones(filter_shape, dtype=filt.dtype) - 5)
        
        kl = tf.reduce_sum(-filt_logstd + tf.exp(filt_logstd)**2/prior_std**2/2)
        kl = tf.identity(kl, name= name+'_' + 'conv_filter_priorkl')

        graph.add_to_collection('kls', kl)

        filt = tf.random_normal(filter_shape)*filt_sd + filt

    x = tf.nn.conv2d(x, filt, [1,1,1,1], 'SAME')

    if use_bias:
        b = tf.get_variable(name + '_' + 'cov_bias', initializer=tf.random_normal([nbfilter], stddev=0.05))
        x += b

    if activation is not None:
        x = activation(x)

    x.set_shape([None,None,None,nbfilter])
    return x 

def clstm(x, n_hidden, filtersize, name, bayesian=False):
    num_channels = int(x.shape[-1])

    with tf.variable_scope('clstm_'+name):
        batchsize, height, width = tf.shape(x)[0], tf.shape(x)[2], tf.shape(x)[3]
        for scope in ['0', '1', '2', '3']:
            with tf.variable_scope(scope):
                cnn(x[:,0], n_hidden, filtersize, '0', bayesian=bayesian)
                cnn(tf.ones([1,30,30,n_hidden]), n_hidden, filtersize, '1', use_bias=False, bayesian=bayesian)

        def step(prev, x):
            st_1, ct_1 = tf.unstack(prev)

            with tf.variable_scope('0', reuse=True):
                tmp0 = cnn(x, n_hidden, filtersize, '0', bayesian=bayesian)
                tmp1 = cnn(st_1, n_hidden, filtersize, '1', use_bias=False, bayesian=bayesian)
            i = tf.sigmoid(tmp0 + tmp1)

            with tf.variable_scope('1', reuse=True):
                tmp0 = cnn(x, n_hidden, filtersize, '0', bayesian=bayesian)
                tmp1 = cnn(st_1, n_hidden, filtersize, '1', use_bias=False, bayesian=bayesian)
            f = tf.sigmoid(tmp0 + tmp1)

            with tf.variable_scope('2', reuse=True):
                tmp0 = cnn(x, n_hidden, filtersize, '0', bayesian=bayesian)
                tmp1 = cnn(st_1, n_hidden, filtersize, '1', use_bias=False, bayesian=bayesian)
            o = tf.sigmoid(tmp0 + tmp1)

            with tf.variable_scope('3', reuse=True):
                tmp0 = cnn(x, n_hidden, filtersize, '0', bayesian=bayesian)
                tmp1 = cnn(st_1, n_hidden, filtersize, '1', use_bias=False, bayesian=bayesian)

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
