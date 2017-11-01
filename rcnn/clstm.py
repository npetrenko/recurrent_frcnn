import tensorflow as tf

eps = 0.001

def create_priorkl(pairs, prior_std):
    with tf.name_scope('KL'):
        for mu, logsigma in pairs:
            kl = -logsigma + (tf.exp(logsigma)**2 + mu**2)/(2*prior_std**2)
            kl = tf.reduce_sum(kl)
            tf.add_to_collection('KLS', kl)

def cnn(x, nbfilter, filtersize, name, use_bias=True, num_channels=None, bayesian=False, prior_std=1., activation=None, reuse=None):
    if num_channels is None:
        num_channels = int(x.shape[-1])

    kernelshape = [filtersize, filtersize, num_channels, nbfilter]

    with tf.variable_scope(name, initializer=tf.random_normal_initializer(stddev=0.01), reuse=reuse):
        with tf.name_scope(name + '/'):

            kernel_mu = tf.get_variable('kernel_mu', shape=kernelshape)

            if bayesian:
                kernel_logsigma = tf.get_variable('kernel_logsigma', shape=kernelshape) - 4
                kernel_sigma = tf.exp(kernel_logsigma)

                tf.summary.histogram('kernel_sigma', kernel_sigma)

                graph = tf.get_default_graph()
                learning_phase = graph.get_tensor_by_name('learning_phase:0')

                pmu = tf.nn.conv2d(x, kernel_mu, [1,1,1,1], padding='SAME')
                pvar = tf.nn.conv2d(x**2, kernel_sigma**2, [1,1,1,1], padding='SAME') + eps

                p = tf.random_normal(tf.shape(pmu))*tf.sqrt(pvar) + pmu

                p = tf.where(learning_phase, p, pmu, name='bayes_learning_phase_switch')

            else:
                p = tf.nn.conv2d(x, kernel_mu, [1,1,1,1], 'SAME')

            if not reuse:
                create_priorkl([[kernel_mu, kernel_logsigma]], prior_std)

            if use_bias:
                b = tf.get_variable('bias', shape=[1,1,1,nbfilter])
                p += b

            if activation is not None:
                p = activation(p)

            p.set_shape([None,None,None,nbfilter])
            return p

def clstm(x, n_hidden, filtersize, name, bayesian=False):
    num_channels = int(x.shape[-1])

    with tf.variable_scope('clstm_'+name):
        batchsize, height, width = tf.shape(x)[0], tf.shape(x)[2], tf.shape(x)[3]
        for scope in ['0', '1', '2', '3']:
            with tf.variable_scope(scope):
                cnn(x[:,0], n_hidden, filtersize, '0', bayesian=bayesian, reuse=False)
                cnn(tf.ones([1,30,30,n_hidden]), n_hidden, filtersize, '1', use_bias=False, bayesian=bayesian, reuse=False)

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
