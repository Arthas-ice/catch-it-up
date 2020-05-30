import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import os
import sys
import time
import pickle
import random
import os
import sys
#from modelarts.session import Session
from utils import *
weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.00001

reduction_ratio = 4

batch_size = 32
iteration = 391
# 128 * 391 ~ 50,000

test_iteration = 10

total_epochs = 40

height=10
width=10
class_num = 10
image_size = 32
img_channels = 1
def load_data(input_dir):

    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")
    input_paths = glob.glob(os.path.join(input_dir, "*.dat"))
    if len(input_paths) == -1:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)
        # 用图片输入明显不合适来了
    with tf.name_scope("load_datas"):
        paths = []
        inputs = []
        data_a = []
        data_b = []
        for demonstration in input_paths:
            temp = []
            with open(demonstration) as f:
                for trajectory in f:
                    s = trajectory.strip().split(' ')
                    temp.append(list(map(int, s)))
                    if len(temp) is height:
                        inputs.append(temp)
                        temp = []
                        if len(inputs) > 1:
                            data_a.append(inputs[-2])
                            data_b.append(extract_action(inputs[-1]))
                            paths.append(str([data_a[-1], data_b[-1]]))

            assert len(temp) is 0
            inputs = []
    print(data_a)
    num_samples = len(paths)
    data_a = np.reshape(data_a, [np.shape(data_a)[0], height, width])
    data_b = np.reshape(data_b, [np.shape(data_b)[0], -1])
    return [data_a, data_b]


def prepare_data():
    print("======Loading data======")
    data_dir ='data0'
    train_data, train_labels = load_data(os.path.join(data_dir, "train"))
    test_data, test_labels = load_data(os.path.join(data_dir, "test"))

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")
    npad = ((11, 11), (11, 11))
    train_data_temp = []
    for i in range(0, len(train_data)):
        train_data_temp.append(np.lib.pad(train_data[i], pad_width=npad,
                                          mode='constant', constant_values=0))
    train_data = np.array(train_data_temp)
    train_data = np.reshape(train_data, (-1, 32, 32, 1))
    test_data_temp = []
    for i in range(0, len(test_data)):
        test_data_temp.append(np.lib.pad(test_data[i], pad_width=npad,
                                         mode='constant', constant_values=0))
    test_data = np.array(test_data_temp)
    test_data = np.reshape(test_data, (-1, 32, 32, 1))

    return train_data, train_labels, test_data, test_labels

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding),(0,0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

#灰度归一化
def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = (x_train - np.mean(x_train)) / np.std(x_train)
    x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    return x_train, x_test


def data_augmentation(batch):
    batch = _random_crop(batch, [image_size, image_size], 4)
    return batch

#resNet+SENet架构模块


def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv", activation=True):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride,
                                   padding=padding)
        if activation:
            network = Relu(network)
        return network


def Fully_connected(x, units=class_num, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=True, units=units)


def Relu(x):
    return tf.nn.relu(x)


def Sigmoid(x):
    return tf.nn.sigmoid(x)


def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')


def Max_pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Dropout(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)



def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 32
    test_iteration=np.shape(test_x)[0]//add
    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration  # average loss
    test_acc /= test_iteration  # average accuracy

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary


class SE_Inception_resnet_v2():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_SEnet(x)

    def Stem(self, x, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=32, kernel=[3, 3], stride=2, padding='VALID', layer_name=scope + '_conv1')
            x = conv_layer(x, filter=32, kernel=[3, 3], padding='VALID', layer_name=scope + '_conv2')
            block_1 = conv_layer(x, filter=64, kernel=[3, 3], layer_name=scope + '_conv3')

            split_max_x = Max_pooling(block_1)
            split_conv_x = conv_layer(block_1, filter=96, kernel=[3, 3], stride=2, padding='VALID',
                                      layer_name=scope + '_split_conv1')
            x = Concatenation([split_max_x, split_conv_x])

            split_conv_x1 = conv_layer(x, filter=64, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x1 = conv_layer(split_conv_x1, filter=96, kernel=[3, 3], padding='VALID',
                                       layer_name=scope + '_split_conv3')

            split_conv_x2 = conv_layer(x, filter=64, kernel=[1, 1], layer_name=scope + '_split_conv4')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[7, 1], layer_name=scope + '_split_conv5')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[1, 7], layer_name=scope + '_split_conv6')
            split_conv_x2 = conv_layer(split_conv_x2, filter=96, kernel=[3, 3], padding='VALID',
                                       layer_name=scope + '_split_conv7')

            x = Concatenation([split_conv_x1, split_conv_x2])

            split_conv_x = conv_layer(x, filter=192, kernel=[3, 3], stride=2, padding='VALID',
                                      layer_name=scope + '_split_conv8')
            split_max_x = Max_pooling(x)

            x = Concatenation([split_conv_x, split_max_x])

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            return x

    def Inception_resnet_A(self, x, scope):
        with tf.name_scope(scope):
            init = x

            split_conv_x1 = conv_layer(x, filter=32, kernel=[1, 1], layer_name=scope + '_split_conv1')

            split_conv_x2 = conv_layer(x, filter=32, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=32, kernel=[3, 3], layer_name=scope + '_split_conv3')

            split_conv_x3 = conv_layer(x, filter=32, kernel=[1, 1], layer_name=scope + '_split_conv4')
            split_conv_x3 = conv_layer(split_conv_x3, filter=48, kernel=[3, 3], layer_name=scope + '_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=64, kernel=[3, 3], layer_name=scope + '_split_conv6')

            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3])
            x = conv_layer(x, filter=384, kernel=[1, 1], layer_name=scope + '_final_conv1', activation=False)

            x = x * 0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            return x

    def Inception_resnet_B(self, x, scope):
        with tf.name_scope(scope):
            init = x

            split_conv_x1 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv1')

            split_conv_x2 = conv_layer(x, filter=128, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=160, kernel=[1, 7], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=192, kernel=[7, 1], layer_name=scope + '_split_conv4')

            x = Concatenation([split_conv_x1, split_conv_x2])
            x = conv_layer(x, filter=1152, kernel=[1, 1], layer_name=scope + '_final_conv1', activation=False)
            # 1154
            x = x * 0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            return x

    def Inception_resnet_C(self, x, scope):
        with tf.name_scope(scope):
            init = x

            split_conv_x1 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv1')

            split_conv_x2 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=224, kernel=[1, 3], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=256, kernel=[3, 1], layer_name=scope + '_split_conv4')

            x = Concatenation([split_conv_x1, split_conv_x2])
            x = conv_layer(x, filter=2144, kernel=[1, 1], layer_name=scope + '_final_conv2', activation=False)
            # 2048
            x = x * 0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            return x

    def Reduction_A(self, x, scope):
        with tf.name_scope(scope):
            k = 256
            l = 256
            m = 384
            n = 384

            split_max_x = Max_pooling(x)

            split_conv_x1 = conv_layer(x, filter=n, kernel=[3, 3], stride=2, padding='VALID',
                                       layer_name=scope + '_split_conv1')

            split_conv_x2 = conv_layer(x, filter=k, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=l, kernel=[3, 3], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=m, kernel=[3, 3], stride=2, padding='VALID',
                                       layer_name=scope + '_split_conv4')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2])

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            return x

    def Reduction_B(self, x, scope):
        with tf.name_scope(scope):
            split_max_x = Max_pooling(x)

            split_conv_x1 = conv_layer(x, filter=256, kernel=[1, 1], layer_name=scope + '_split_conv1')
            split_conv_x1 = conv_layer(split_conv_x1, filter=384, kernel=[3, 3], stride=2, padding='VALID',
                                       layer_name=scope + '_split_conv2')

            split_conv_x2 = conv_layer(x, filter=256, kernel=[1, 1], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=288, kernel=[3, 3], stride=2, padding='VALID',
                                       layer_name=scope + '_split_conv4')

            split_conv_x3 = conv_layer(x, filter=256, kernel=[1, 1], layer_name=scope + '_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=288, kernel=[3, 3], layer_name=scope + '_split_conv6')
            split_conv_x3 = conv_layer(split_conv_x3, filter=320, kernel=[3, 3], stride=2, padding='VALID',
                                       layer_name=scope + '_split_conv7')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3])

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            return x

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation

            return scale

    def Build_SEnet(self, input_x):
        input_x = tf.pad(input_x, [[0, 0], [32, 32], [32, 32],[0,0]])
        # size 32 -> 96
        print(np.shape(input_x))
        # only cifar10 architecture

        x = self.Stem(input_x, scope='stem')

        for i in range(5):
            x = self.Inception_resnet_A(x, scope='Inception_A' + str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A' + str(i))

        x = self.Reduction_A(x, scope='Reduction_A')

        channel = int(np.shape(x)[-1])
        x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A')

        for i in range(10):
            x = self.Inception_resnet_B(x, scope='Inception_B' + str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B' + str(i))

        x = self.Reduction_B(x, scope='Reduction_B')

        channel = int(np.shape(x)[-1])
        x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B')

        for i in range(5):
            x = self.Inception_resnet_C(x, scope='Inception_C' + str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C' + str(i))

        # channel = int(np.shape(x)[-1])
        # x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C')

        x = Global_Average_Pooling(x)
        x = Dropout(x, rate=0.2, training=self.training)
        x = flatten(x)

        x = Fully_connected(x, layer_name='final_fully_connected')
        return x


def compute_loss(inputs, outputs,targets):
        with tf.variable_scope("ilpo_loss"):
            out_channels = int(targets.get_shape()[-1])
            # compute loss on expected next state.
            delta = slim.flatten(targets - extract_contrast(inputs))
            gen_loss_exp = tf.reduce_mean(
                tf.reduce_sum(tf.losses.mean_squared_error(delta, slim.flatten(expected_outputs),
                                                   reduction=tf.losses.Reduction.NONE), axis=1))

            # compute loss on min next state.
            all_loss = []

            for out in outputs:
                all_loss.append(tf.reduce_sum(
                    tf.losses.mean_squared_error(delta, slim.flatten(out),
                    reduction=tf.losses.Reduction.NONE),
                    axis=1))

            stacked_min_loss = tf.stack(all_loss, axis=-1)
            gen_loss_min = tf.reduce_mean(tf.reduce_min(stacked_min_loss, axis=1))
            gen_loss_L1 = gen_loss_exp + gen_loss_min



train_x, train_y, test_x, test_y = prepare_data()
train_x, test_x = color_preprocessing(train_x, test_x)
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])
training_flag = tf.placeholder(tf.bool)

learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = SE_Inception_resnet_v2(x, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("loading..")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('./logs', sess.graph)

    epoch_learning_rate = init_learning_rate
    for epoch in range(1, total_epochs + 1):
        if epoch % 30 == 0:
            epoch_learning_rate = epoch_learning_rate / 10

        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0
        start_time=time.time()
        print("\n epoch %d/%d:" % (epoch, total_epochs))
        iteration=np.shape(train_x)[0]//batch_size
        for step in range(1, iteration + 1):
            if pre_index + batch_size < 60000:
                batch_x = train_x[pre_index: pre_index + batch_size]
                batch_y = train_y[pre_index: pre_index + batch_size]
            else:
                batch_x = train_x[pre_index:]
                batch_y = train_y[pre_index:]

            batch_x = data_augmentation(batch_x)
            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }

            _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)

            train_loss += batch_loss
            train_acc += batch_acc
            pre_index += batch_size
            if step !=iteration:
               print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f"
                   % (step, iteration, train_loss / step, train_acc / step))
        train_loss /= iteration  # average loss
        train_acc /= iteration  # average accuracy

        train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                          tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

        test_acc, test_loss, test_summary = Evaluate(sess)

        summary_writer.add_summary(summary=train_summary, global_step=epoch)
        summary_writer.add_summary(summary=test_summary, global_step=epoch)
        summary_writer.flush()
        line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
            epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
        print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
              "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
              % (iteration, iteration, int(time.time() - start_time), train_loss, train_acc, test_loss, test_acc))
        with open('./logs.txt', 'a') as f:
            f.write(line)
        saver.save(sess=sess, save_path='./model/Inception_resnet_v2.ckpt')
