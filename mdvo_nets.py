# The network design is based on Tinghui Zhou works:
# https://github.com/tinghuiz/SfMLearner/blob/master/nets.py
from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from gan_function import discrim_conv_cgan, lrelu_cgan, batchnorm_cgan

# Range of disparity/inverse depth values
def T_Refine_Net(opt, posenet_inputs):
    bs = opt.batch_size
    is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('train_T_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(posenet_inputs, 16, 7, 2)
            conv2 = slim.conv2d(conv1, 32, 5, 2)
            conv3 = slim.conv2d(conv2, 64, 3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose_pred = slim.conv2d(conv7, 3 * (opt.num_source), 1, 1, normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            translation_refine = 0.01 * tf.reshape(pose_avg, [-1, (opt.num_source), 3])
            temp = tf.constant([[[0., 0., 0.]]])
            temp = tf.tile(temp, [bs, 1, 1])
            translation_refine = tf.concat([translation_refine, temp], axis=2)  # (4, 1, 6) only one pose
            return translation_refine

def TT_Refine_Net(opt, posenet_inputs):
    bs = opt.batch_size
    is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('train_TT_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(posenet_inputs, 16, 7, 2)
            conv2 = slim.conv2d(conv1, 32, 5, 2)
            conv3 = slim.conv2d(conv2, 64, 3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose_pred = slim.conv2d(conv7, 3 * (opt.num_source), 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            translation_refine = 0.01 * tf.reshape(pose_avg, [-1, opt.num_source, 3])
            temp = tf.constant([[[0., 0., 0.]]])
            temp = tf.tile(temp, [bs, 1, 1])
            translation_refine = tf.concat([translation_refine, temp], axis=2)
            return translation_refine

def TTT_Refine_Net(opt, posenet_inputs):
    bs = opt.batch_size
    is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('train_TTT_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(posenet_inputs, 16, 7, 2)
            conv2 = slim.conv2d(conv1, 32, 5, 2)
            conv3 = slim.conv2d(conv2, 64, 3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose_pred = slim.conv2d(conv7, 3 * (opt.num_source), 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            translation_refine = 0.01 * tf.reshape(pose_avg, [-1, (opt.num_source), 3])
            temp = tf.constant([[[0., 0., 0.]]])
            temp = tf.tile(temp, [bs, 1, 1])
            translation_refine = tf.concat([translation_refine, temp], axis=2)
            return translation_refine

def TTTT_Refine_Net(opt, posenet_inputs):
    bs = opt.batch_size
    is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('train_TTTT_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(posenet_inputs, 16, 7, 2)
            conv2 = slim.conv2d(conv1, 32, 5, 2)
            conv3 = slim.conv2d(conv2, 64, 3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose_pred = slim.conv2d(conv7, 3 * (opt.num_source), 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            translation_refine = 0.01 * tf.reshape(pose_avg, [-1, (opt.num_source), 3])
            temp = tf.constant([[[0., 0., 0.]]])
            temp = tf.tile(temp, [bs, 1, 1])
            translation_refine = tf.concat([translation_refine, temp], axis=2)
            return translation_refine

def R_Refine_Net(opt, posenet_inputs):
    bs = opt.batch_size
    is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('train_R_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(posenet_inputs, 16, 7, 2)
            conv2 = slim.conv2d(conv1, 32, 5, 2)
            conv3 = slim.conv2d(conv2, 64, 3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose_pred = slim.conv2d(conv7, 3 * opt.num_source, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            rotation_refine = 0.01 * tf.reshape(pose_avg, [-1, opt.num_source, 3])
            temp = tf.constant([[[0., 0., 0.]]])
            temp = tf.tile(temp, [bs, 1, 1])
            rotation_refine = tf.concat([temp, rotation_refine], axis=2)
            return rotation_refine


def RR_Refine_Net(opt, posenet_inputs):
    bs = opt.batch_size
    is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('train_RR_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(posenet_inputs, 16, 7, 2)
            conv2 = slim.conv2d(conv1, 32, 5, 2)
            conv3 = slim.conv2d(conv2, 64, 3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose_pred = slim.conv2d(conv7, 3 * opt.num_source, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            rotation_refine = 0.01 * tf.reshape(pose_avg, [-1, opt.num_source, 3])
            temp = tf.constant([[[0., 0., 0.]]])
            temp = tf.tile(temp, [bs, 1, 1])
            rotation_refine = tf.concat([temp, rotation_refine], axis=2)
            return rotation_refine

def RRR_Refine_Net(opt, posenet_inputs):
    bs = opt.batch_size
    is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('train_RRR_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(posenet_inputs, 16, 7, 2)
            conv2 = slim.conv2d(conv1, 32, 5, 2)
            conv3 = slim.conv2d(conv2, 64, 3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose_pred = slim.conv2d(conv7, 3 * opt.num_source, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            rotation_refine = 0.01 * tf.reshape(pose_avg, [-1, opt.num_source, 3])
            temp = tf.constant([[[0., 0., 0.]]])
            temp = tf.tile(temp, [bs, 1, 1])
            rotation_refine = tf.concat([temp, rotation_refine], axis=2)
            return rotation_refine

def RRRR_Refine_Net(opt, posenet_inputs):
    bs = opt.batch_size
    is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('train_RRRR_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(posenet_inputs, 16, 7, 2)
            conv2 = slim.conv2d(conv1, 32, 5, 2)
            conv3 = slim.conv2d(conv2, 64, 3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose_pred = slim.conv2d(conv7, 3 * opt.num_source, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            rotation_refine = 0.01 * tf.reshape(pose_avg, [-1, opt.num_source, 3])
            temp = tf.constant([[[0., 0., 0.]]])
            temp = tf.tile(temp, [bs, 1, 1])
            rotation_refine = tf.concat([temp, rotation_refine], axis=2)
            return rotation_refine

def discriminate_net_2(opt, ori_inputs, inputs, reuse):
    DIM = 64
    with tf.variable_scope('D_net_2', reuse=reuse):
            n_layers = 3
            layers = []
            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input = tf.concat([ori_inputs, inputs], axis=3)

            # layer_1: [batch, 128, 416, in_channels * 2] => [batch, 64, 208, DIM]
            with tf.variable_scope("layer_1"):
                convolved = discrim_conv_cgan(input, DIM, stride=2)
                rectified = lrelu_cgan(convolved, 0.2)
                layers.append(rectified)
            # layer_2: [batch, 64, 208, DIM] => [batch, 32, 104, DIM * 2]
            # layer_3: [batch, 32, 104, DIM * 2] => [batch, 16, 52, DIM * 4]
            # layer_4: [batch, 16, 52, DIM * 4] => [batch, 15, 51, DIM * 8]
            for i in range(n_layers):
                with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                    out_channels = DIM * min(2 ** (i + 1), 8)
                    stride = 1 if i == n_layers - 1 else 2
                    convolved = discrim_conv_cgan(layers[-1], out_channels, stride=stride)
                    normalized = batchnorm_cgan(convolved)
                    rectified = lrelu_cgan(normalized, 0.2)
                    layers.append(rectified)
            # layer_5: [batch, 15, 51, ndf * 8] => [batch, 14, 50, 1]
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = discrim_conv_cgan(rectified, out_channels=1, stride=1)
                output = tf.sigmoid(convolved)
                layers.append(output)

            return layers[-1]

