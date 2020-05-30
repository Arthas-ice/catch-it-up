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
import glob
weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.01

reduction_ratio = 4

batch_size = 128
iteration = 391
# 128 * 391 ~ 50,000

test_iteration = 10

total_epochs = 40

height=16
width=16
class_num = 16
image_size = 32
img_channels = 1

def extract_action( output_data):
    return output_data[-1]

def prepare_data(data_path,id):
    if data_path is None or not os.path.exists(data_path):
        raise Exception("input_dir does not exist")
    start_time = time.time()
    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name
    while 1:
        input_paths = glob.glob(os.path.join(data_path, "*.dat"))
        if len(input_paths) == -1:
            raise Exception("input_dir contains no image files")
        if all(get_name(path).isdigit() for path in input_paths):
            input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
        else:
            input_paths = sorted(input_paths)
        if int(get_name(input_paths[-1]))==id:
            break
        if time.time()-start_time>60:
            raise TimeoutError
    print("receive the data")
    input_path=input_paths[-1]
    with tf.name_scope("load_datas"):
        data = []
        label = []
        temp = []
        with open(input_path) as f:
            for trajectory in f:
                s = trajectory.strip().split(' ')
                temp.append(list(map(int, s)))
        data.append(temp)
        label.append(extract_action(temp))
    npad = ((8, 8), (8, 8))
    data_temp = []
    for i in range(0, len(data)):
        data_temp.append(np.lib.pad(data[i], pad_width=npad,
                                          mode='constant', constant_values=0))
    train_data = np.array(data_temp)
    data = np.reshape(train_data, (-1, 32, 32, 1))
    return [data,label]
def color_preprocessing(x_train):
    x_train = x_train.astype('float32')
    x_train = (x_train - np.mean(x_train)) / np.std(x_train)
    return x_train

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


x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])
training_flag = tf.placeholder(tf.bool)

learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = SE_Inception_resnet_v2(x, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)
out=tf.argmax(logits,1)
#把一个只反应一步的C++加进来就好
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:

    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("loading..")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No model found.")
        raise ModuleNotFoundError
    id=0
    while id<1000:
        print("step: ", id)
        os.chdir("./")
        path_01 = r"rob.exe %s" % (str(id))
        r_v = os.system(path_01)
        id+=1
        test_data, test_label= prepare_data( "./table",id)
        test_data=color_preprocessing(test_data)
        test_data=test_data[0:]
        test_label=test_label[0:]
        feed_dict={
            x: test_data,
            label: test_label,
            learning_rate: 0.01,
            training_flag: False
        }
        output=sess.run([out],feed_dict=feed_dict)
        with open("./action/"+str(id)+".dat", "w") as f:
            f.write(str(output[0].tolist()[0]))
    print("finished")


