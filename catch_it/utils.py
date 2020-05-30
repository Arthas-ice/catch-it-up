from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#这种格式的参数设定，妙啊
#但是这个可能不太方便
#TODO:将args改为内置参数
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import tensorflow as tf
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="train", help="path to folder containing images")
parser.add_argument("--mode", default="test", choices=["train", "test", "export"])
parser.add_argument("--output_dir", default="dqn_output", help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--hres", action="store_true")
parser.add_argument("--highd", action="store_false")
parser.add_argument("--num-levels", type=int)
parser.add_argument("--set-seed", type=int)
parser.add_argument("--checkpoint", help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--exp_dir", help="directory for saving experiments")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=5000, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=32, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=128, help="number of generator filters in first encoding layer")
parser.add_argument("--ndf", type=int, default=128, help="number of discriminator filters in first encoding layer")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=False)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--policy_lr", type=float, default=0.002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--n_actions", type=int, help="The number of generators to predict")
parser.add_argument("--real_actions", type=int, help="The real number of actions to predict")
parser.add_argument("--n_dims", type=int, help="The number of dimensions to predict")
parser.add_argument("--env", help="The name of the environment")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

#新的参数，确定了游戏的长宽
parser.add_argument("--height", type=int, default=10)
parser.add_argument("--width", type=int, default=10)
parser.add_argument("--max_x", type=int, default=3)
parser.add_argument("--test_dir",default='test')

args = parser.parse_args()

EPS = 1e-12

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch,input_actions")
Model = collections.namedtuple(
     "Model", "outputs, expectation, min_output, actions, gen_loss_L1, gen_grads_and_vars, train, train_acc, test_acc")
BCModel = collections.namedtuple(
     "Model", "actions, gen_loss_L1, gen_grads_and_vars, train")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def fully_connected(inputs, n_outputs, reuse=False, scope=None):
    inputs = slim.flatten(inputs)

    return slim.fully_connected(inputs, n_outputs, activation_fn=None, reuse=reuse, scope=scope)

    with tf.variable_scope("fc"):
        w_fc = weight_variable([int(inputs.shape[-1]), n_outputs])
        b_fc = bias_variable([n_outputs])
        return tf.matmul(inputs, w_fc) + b_fc


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        print(filter)
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch = tf.shape(batch_input)[0]
        in_height = int(batch_input.shape[1])
        in_width = int(batch_input.shape[2])
        in_channels = int(batch_input.shape[3])
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def save_images(fetches, step=None):
    image_dir = os.path.join(args.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(args.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

def extract_action( output_data):
    return output_data[-1]


