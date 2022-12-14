from __future__ import division
import os
import time
import random
import pprint
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from mdvo_test_depth import *
from mdvo_test_pose import *
from data_loader import DataLoader

flags = tf.app.flags
flags.DEFINE_string("mode",                         "",    "(train_rigid, train_flow) or (test_depth, test_pose, test_flow)")
flags.DEFINE_string("dataset_dir",                  "",    "Dataset directory")
flags.DEFINE_string("dataset_dir_test_odom",        "",    "Dataset directory of odom")
flags.DEFINE_string('gt_2015_dir',                  '',    'directory of ground truth of kitti 2015')
flags.DEFINE_string("pred_mask",                    "",    "chose directory to save pred mask")
flags.DEFINE_string("store_model",                  "",    "according loss to store model")
flags.DEFINE_string("init_ckpt_file",             None,    "Specific checkpoint file to initialize from")
flags.DEFINE_integer("batch_size",                   4,    "The size of of a sample batch")
flags.DEFINE_integer("num_threads",                 32,    "Number of threads for data loading")
flags.DEFINE_integer("img_height",                 128,    "Image height")
flags.DEFINE_integer("img_width",                  416,    "Image width")
flags.DEFINE_integer("seq_length",                   2,    "Sequence length for each example")

# #### Training Configurations #####
flags.DEFINE_string("checkpoint_dir",               "",    "Directory name to save the checkpoints")
flags.DEFINE_float("learning_rate",             0.0001,    "Learning rate for adam")
flags.DEFINE_integer("max_to_keep",                100,    "Maximum number of checkpoints to save")
flags.DEFINE_integer("max_steps",               350100,    "Maximum number of training iterations")
flags.DEFINE_integer("save_ckpt_freq",            2500,    "Save the checkpoint model every save_ckpt_freq iterations")
flags.DEFINE_float("alpha_recon_image",           0.85,    "Alpha weight between SSIM and L1 in reconstruction loss")
flags.DEFINE_boolean("num_source",                  None,    "pose number")
flags.DEFINE_boolean("add_dispnet",                 None,    "add disp net")
flags.DEFINE_boolean("add_posenet",                 None,    "add pose net")
flags.DEFINE_boolean("add_flownet",                 None,    "add flow net")
flags.DEFINE_boolean("num_scales",                  None,     "scales number")

# #### Configurations about DepthNet & PoseNet  #####
flags.DEFINE_float("rigid_warp_weight",            1.0,    "Weight for warping by rigid flow")
flags.DEFINE_float("disp_smooth_weight",           0.5,    "Weight for disp smoothness")

flags.DEFINE_float("flow_warp_weight",             1.0,    "Weight for warping by full flow")
flags.DEFINE_float("flow_smooth_weight",           10.0,    "Weight for flow smoothness")
flags.DEFINE_float("flow_consistency_weight",      0.01,    "Weight for bidirectional flow consistency")
flags.DEFINE_float("flow_consistency_alpha",       3.0,    "Alpha for flow consistency check")
flags.DEFINE_float("flow_consistency_beta",        0.05,    "Beta for flow consistency check")
flags.DEFINE_float("ssim_weight",  0.85, "Weight for using ssim loss in pixel loss")
flags.DEFINE_float("image_loss",   0.85, "Weight for using image loss in pixel loss")

# #### Testing Configurations #####
flags.DEFINE_string("output_dir",                None,     "Test result output directory")
flags.DEFINE_string("depth_test_split",         "stereo",  "KITTI depth split, eigen or stereo")
flags.DEFINE_integer("pose_test_seq",            None,      "KITTI Odometry Sequence ID to test")
flags.DEFINE_string("summary_out_dir",           None,     "the dir is to save summary for tensorboard show")
# flow
flags.DEFINE_float("flow_diff_threshold",        4.0,     "threshold when comparing optical flow and rigid flow ")
flags.DEFINE_boolean("add_gan",                  True,    "Choice to use gan, default is True")

opt = flags.FLAGS

def train():
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    if not os.path.exists(opt.summary_out_dir):
        os.makedirs(opt.summary_out_dir)

    with tf.Graph().as_default():
        # Data Loader
        loader = DataLoader(opt)
        tgt_image_left, src_image_left, intrinsics_left, tgt_image_right, src_right_image = loader.load_train_batch()

        # Build Model
        model = RA_GANVO_Model(opt, tgt_image_left, src_image_left, intrinsics_left, tgt_image_right, src_right_image)
        rigid_loss = model.rigid_loss
        d_loss = model.d_loss_finall
        g_loss = model.g_loss_finall
        g_loss_1 = model.g_loss

        summary_op = tf.summary.merge_all('summary_see')

        if opt.mode == 'train_rigid':
            vars = [var for var in tf.trainable_variables()]
            train_vars = [var for var in vars if "train_" in var.name]    # G
            train_vars_d = [var for var in vars if "D_net" in var.name]   # D
            vars_to_restore = slim.get_variables_to_restore()
        else:
            train_vars = [var for var in tf.trainable_variables()]

        if opt.init_ckpt_file != None:
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(opt.init_ckpt_file, vars_to_restore)

        ##########################
        # # Global Step
        optim = tf.train.AdamOptimizer(opt.learning_rate, 0.9)
        optim_d = tf.train.AdamOptimizer(opt.learning_rate, 0.9)

        train_op = slim.learning.create_train_op(g_loss, optim, variables_to_train=train_vars)
        train_op_d = slim.learning.create_train_op(d_loss, optim_d, variables_to_train=train_vars_d)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        incr_global_step = tf.assign(global_step, global_step + 1)

        # Parameter Count
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in vars])

        # in order to continue to train
        saver = tf.train.Saver(max_to_keep=opt.max_to_keep)

        # Session
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, save_summaries_secs=None, saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in vars:
                print(var.name)
            if opt.add_gan:
                for var_d in train_vars_d:
                    print(var_d.name)
            print("parameter_count =", sess.run(parameter_count))

            if opt.init_ckpt_file != None:
                sess.run(init_assign_op, init_feed_dict)

            start_time = time.time()

            summary_writer = tf.summary.FileWriter(opt.summary_out_dir, sess.graph)
            for step in range(1, opt.max_steps):
                # firs train D, then train G

                fetches_d = {"train": train_op_d}
                sess.run(fetches_d)

                fetches = {"train": train_op,
                           "global_step": global_step,
                           "incr_global_step": incr_global_step}

                if step % 100 == 0:
                    fetches["rigid_loss"] = rigid_loss
                    fetches["d_loss"] = d_loss
                    fetches["g_loss"] = g_loss_1

                results = sess.run(fetches)
                if step % 100 == 0:
                    time_per_iter = (time.time() - start_time) / 100
                    start_time = time.time()
                    print('Iteration: [%7d] | Time: %4.4fs/iter|rigid_loss:%.6f| g_loss:%.6f |d_loss:%.14f'
                          % (step, time_per_iter, results["rigid_loss"], results["g_loss"], results["d_loss"]))

                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, global_step=step)
                if step % opt.save_ckpt_freq == 0:
                    saver.save(sess, os.path.join(opt.checkpoint_dir, 'model'), global_step=step)

def main(_):
    opt.num_source = opt.seq_length - 1
    opt.num_scales = 4
    opt.add_flownet = opt.mode in ['train_flow', 'test_flow']
    opt.add_dispnet = opt.mode in ['train_rigid', 'test_depth', 'test_pose']
    opt.add_posenet = opt.mode in ['train_rigid', 'test_pose']

    if opt.mode in ['train_rigid', 'train_flow']:
        train()
    elif opt.mode == 'test_depth':
        test_depth(opt)
    elif opt.mode == 'test_pose':
        test_pose(opt)

if __name__ == '__main__':
    tf.app.run()


