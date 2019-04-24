import numpy as np


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)


class RunningBatchNorm(object):

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-8  # init with non-zero value due to division

    def update(self, batch):
        """Update running statistics from moments computed on the collected batch.
        The variance is computed with the algorithm described at the following:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        # Compute statistics from minibatch
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        # Assemble extra statistics
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        # Compute moments
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        # Update statistics
        self.mean = self.mean + delta * batch_count / total_count
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, x):
        """Use the computed running statistics to normalize the input `x`"""
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=()):

        self._sum = tf.get_variable(
            dtype=tf.float64,
            shape=shape,
            initializer=tf.constant_initializer(0.0),
            name="runningsum", trainable=False)
        self._sumsq = tf.get_variable(
            dtype=tf.float64,
            shape=shape,
            initializer=tf.constant_initializer(epsilon),
            name="runningsumsq", trainable=False)
        self._count = tf.get_variable(
            dtype=tf.float64,
            shape=(),
            initializer=tf.constant_initializer(epsilon),
            name="count", trainable=False)
        self.shape = shape

        self.mean = tf.to_float(self._sum / self._count)
        self.std = tf.sqrt( tf.maximum( tf.to_float(self._sumsq / self._count) - tf.square(self.mean) , 1e-2 ))

        newsum = tf.placeholder(shape=self.shape, dtype=tf.float64, name='sum')
        newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float64, name='var')
        newcount = tf.placeholder(shape=[], dtype=tf.float64, name='count')
        self.incfiltparams = U.function([newsum, newsumsq, newcount], [],
            updates=[tf.assign_add(self._sum, newsum),
                     tf.assign_add(self._sumsq, newsumsq),
                     tf.assign_add(self._count, newcount)])


    def update(self, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n*2+1, 'float64')
        addvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)],dtype='float64')])
        if MPI is not None:
            MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        self.incfiltparams(totalvec[0:n].reshape(self.shape), totalvec[n:2*n].reshape(self.shape), totalvec[2*n])
