import tensorflow as tf
import functools
import matplotlib.pyplot as plt
import numpy as np
from collections import Iterable


# Global
sess = None
_coord, _threads = None, None


# aliases
tfloat = tf.float32
tint = tf.int32
nfloat = np.float32
var = tf.Variable
add = tf.add
mul = tf.matmul
sqr = tf.square
equal = tf.equal
relu = tf.nn.relu
sigmoid = tf.nn.sigmoid
softmax = tf.nn.softmax # 전체 합을 1로 만듦(확률)
argmax = tf.argmax  # 가장 큰 값의 인덱스
rmean = tf.reduce_mean
rsum = tf.reduce_sum
cast = tf.cast # T/F -> 1/0
log = tf.log
array = np.array
transpose = np.transpose # 행/열 -> 열/행
ckpt_stat = tf.train.get_checkpoint_state
ckpt_exists = tf.train.checkpoint_exists
name_scope = tf.name_scope
var_scope = tf.variable_scope
layer = tf.layers.dense
merge_all = tf.summary.merge_all
dropout = tf.nn.dropout
rnncell = tf.nn.rnn_cell.BasicRNNCell
lstmcell = tf.nn.rnn_cell.BasicLSTMCell
multicell = tf.nn.rnn_cell.MultiRNNCell
str_producer = tf.train.string_input_producer
batch = tf.train.batch
one_hot = tf.one_hot
histo = tf.summary.histogram
scalar = tf.summary.scalar
getvar = tf.get_variable


def varxavier(name, shape=None):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())


def ph(shape=None, name=None, dt=tfloat):
    return tf.placeholder(dtype=dt, shape=shape, name=name)


def varnorm(shape, stddev=None, name=None):
    if stddev:
        return var(tf.random_normal(shape, stddev=stddev), name=name)
    else:
        return var(tf.random_normal(shape), name=name) # defeault 1.0


def varuni(shape, minval=-1.0, maxval=1.0, dtype=tfloat, name=None):
    return var(tf.random_uniform(shape, minval, maxval, dtype=dtype), name=name)


def varzeros(shape):
    return var(tf.zeros(shape))


def fcomp(*funs):
    def inner(*args, **kwargs):
        res = funs[-1](*args, **kwargs)
        for f in reversed(funs[:-1]):
            res = f(res)

    return inner


def reduce(funs, *args):
    if not isinstance(funs, Iterable):
        funcs = [funs]

    it = iter(funs)
    return functools.reduce(lambda x, y: next(it, funs[0])(x, y), args)


# if queue_runner, you must call close()
def open(restore=False, ckpt_dir=None, queue_runner=False):
    global sess, _coord, _threads
    sess, saver = tf.Session(), None

    if not restore:
        sess.run(tf.global_variables_initializer())
    else:
        saver = tf.train.Saver(tf.global_variables())

        assert ckpt_dir != None ,"Set checkpoint directory"
        ckpt = ckpt_stat(ckpt_dir)
        if ckpt and ckpt_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

    if queue_runner:
        _coord = tf.train.Coordinator()
        _threads = tf.train.start_queue_runners(sess, _coord)

    return sess, saver


def close():
    _coord.request_stop()
    _coord.join()


def run(fetches, dic=None, opt=None, mrun=None):
    return sess.run(fetches, dic, opt, mrun) if sess else None


def opt_grd(learning_rate=0.1):
    return tf.train.GradientDescentOptimizer(learning_rate)


def opt_adam(learning_rate=0.1):
    return tf.train.AdamOptimizer(learning_rate)


def opt_rms(learning_rate=0.1):
    return tf.train.RMSPropOptimizer(learning_rate)


def cross_entropy(labels, logits):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)


def sigmoid_cross_entropy(labels, logits):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)


def load_csv(fname):
    return np.loadtxt(fname, delimiter=',', dtype='float32')
    #return np.loadtxt(fname, delimiter=',', unpack=True, dtype='float32') # unpack -> transposed


def read_csvfiles(filenames, rec):
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False, name='filename_queue')
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    data = tf.decode_csv(value, record_defaults=rec)
    return data


def file_writer(log_dir):
    return tf.summary.FileWriter(log_dir, sess.graph) if sess else None


#collect = functools.partial(tf.get_collection, key=tf.GraphKeys.TRAINABLE_VARIABLES)
def collect(scope, key=tf.GraphKeys.TRAINABLE_VARIABLES):
    return tf.get_collection(key, scope)


def rnn(cell, input):
    return tf.nn.dynamic_rnn(cell, input ,dtype=tfloat)


def accuracy(p, label, dic):
    acc = rmean(cast(equal(p, label), tfloat))
    return run(acc * 100, dic)

