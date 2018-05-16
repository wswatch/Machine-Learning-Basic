import os
import re
import sys
import tarfile
import math
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import sys

from six.moves import urllib
import tensorflow as tf

File_directory = "./fire"
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = File_directory
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

maybe_download_and_extract()


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def build_cnn():
    _SIZE = 32
    _NUM_CLASSES = 10

    with tf.name_scope('main_params'):
        x = tf.placeholder(tf.float32, shape=[None, _SIZE,  _SIZE, 3], name='Input_Image')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Result')
        global_step = tf.Variable(initial_value=0, trainable=True, name='Global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='Learning_rate')

    with tf.variable_scope('Layer1') as scope:
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=[3,3],
            padding='SAME',
            activation=tf.nn.relu
        )
    with tf.variable_scope('Layer2') as scope:
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=[3,3],
            padding='SAME',
            activation=tf.nn.relu
        )
    with tf.variable_scope('Layer3') as scope:
        pool1 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2)

    with tf.variable_scope('Layer4') as scope:
        conv3 = tf.layers.conv2d(
            inputs=pool1,
            filters=128,
            kernel_size=[3,3],
            activation=tf.nn.relu
        )
    with tf.variable_scope('Layer5') as scope:
        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=128,
            kernel_size=[3,3],
            activation=tf.nn.relu
        )
    with tf.variable_scope('Layer6') as scope:
        pool2 = tf.layers.max_pooling2d(conv4, pool_size=[2,2], strides=2)

    with tf.variable_scope('Layer7') as scope:
        conv5 = tf.layers.conv2d(
            inputs=pool2,
            filters=256,
            kernel_size=[3,3],
            activation=tf.nn.relu
        )
    with tf.variable_scope('Layer8') as scope:
        pool3 = tf.layers.max_pooling2d(conv5, pool_size=[2,2], strides=2)

    with tf.variable_scope('Layer9_10') as scope:
        size = pool3.get_shape().as_list()
        flat = tf.reshape(pool3, [-1, size[1]*size[2]*size[3]])

        fc = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)
        softmax = tf.layers.dense(inputs=fc, units=_NUM_CLASSES, activation=tf.nn.softmax, name=scope.name)
    y_pred_cls = tf.argmax(softmax, axis=1)
    return x, y, softmax, y_pred_cls, global_step, learning_rate



def get_data_set(height, width, num_channel):
    tmp = []
    for i in range(1,6):
        tmp.append(unpickle('./fire/cifar-10-batches-py/data_batch_'+str(i))[b'data'])
    tmp = np.asarray(tmp)
    xtrain_batch_all = np.vstack(tmp)
    test_batch = unpickle('./fire/cifar-10-batches-py/test_batch')
    xtest_batch_all = test_batch[b'data']
    X_train = np.asarray([image.reshape(height, width, num_channel) for image in xtrain_batch_all]) / 255.0
    X_test = np.asarray([image.reshape(height, width, num_channel) for image in xtest_batch_all]) / 255.0
    tmp = unpickle('./fire/cifar-10-batches-py/data_batch_1')[b'labels']
    for i in range(2,6):
        tmp += unpickle('./fire/cifar-10-batches-py/data_batch_' + str(i))[b'labels']
    y_train = np.asarray(tmp)
    y_test = np.asarray(test_batch[b'labels'])
    lb = LabelBinarizer()
    lb.fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=7)
    return X_train, y_train, X_val, y_val, X_test, y_test


def train(train_x, train_y):

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
                                       beta1=0.9,
                                       beta2=0.99,
                                       epsilon=1e-08).minimize(loss, global_step=global_step)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))

    for i in range(_EPOCH):
        print("\nEpoch: {0}/{1}\n".format((i + 1), _EPOCH))
        for s in range(batch_size):
            batch_xs = train_x[s * _BATCH_SIZE: (s + 1) * _BATCH_SIZE]
            batch_ys = train_y[s * _BATCH_SIZE: (s + 1) * _BATCH_SIZE]
            _, _, batch_loss, batch_acc = sess.run(
                [global_step,optimizer, loss, accuracy],
                feed_dict={x: batch_xs, y:batch_ys, learning_rate: 0.0001})
            if s % 100 == 0 :
                print('({} / {}) The loss is {}, get accuracy:{}'.format(s, batch_size, batch_loss, batch_acc))
    saver.save(sess, _SAVE_PATH)



def test(test_x, test_y):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, _SAVE_PATH)
        batch_size = int(math.ceil(len(test_x) / _BATCH_SIZE))
        total_acc = 0
        for s in range(batch_size):
            batch_xs = test_x[s * _BATCH_SIZE: (s + 1) * _BATCH_SIZE]
            batch_ys = test_y[s * _BATCH_SIZE: (s + 1) * _BATCH_SIZE]
            accuracy_test = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            #print('accuracy {}'.format(accuracy_test))
            total_acc += accuracy_test
        print('Test accuracy:{}'.format(total_acc/batch_size))


_BATCH_SIZE = 128
_EPOCH = 3
_SAVE_PATH = "./check-point/checkpoint.ckpt"
x, y, output, y_pred_cls, global_step, learning_rate = build_cnn()
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():
    train_x, train_y, val_x, val_y, test_x, test_y = get_data_set(32, 32, 3)
    if sys.argv[-1] == '-train':
        train(train_x, train_y)
    elif sys.argv[-1] == '-test':
        test(test_x, test_y)
    else:
        raise ValueError('No Command.')

    return

main()
