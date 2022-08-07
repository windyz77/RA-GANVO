from __future__ import division
import os
import math
import scipy.misc as sm
import tensorflow as tf
import numpy as np
from glob import glob
from mdvo_model import *
from data_loader import DataLoader
from kitti_eval.pose_evaluation_utils import *
from offical_test import *
from kitti_eval.eval_pose import eval_pose

def test_pose(opt):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
        opt.img_height, opt.img_width, opt.seq_length * 3],
        name='raw_input')
    input_uint8_r = tf.placeholder(tf.uint8, [opt.batch_size,
                opt.img_height, opt.img_width, opt.seq_length * 3],
                                 name='raw_input_r')
    tgt_image = input_uint8[:, :, :, :3]
    tgt_image_r = input_uint8_r[:, :, :, :3]

    src_image_stack = input_uint8[:, :, :, 3:]
    src_image_stack_r = input_uint8_r[:, :, :, 3:]

    intrinsics = tf.placeholder(tf.float32, [opt.batch_size, 3, 3], name='intrinsics_input')

    loader = DataLoader(opt)
    intrinsics_ms = loader.get_multi_scale_intrinsics(intrinsics, opt.num_scales)
    model = RA_GANVO_Model(opt, tgt_image, src_image_stack, intrinsics_ms, tgt_image_r, src_image_stack_r)
    fetches = {"pose": model.pred_rt_full}

    saver = tf.train.Saver([var for var in tf.model_variables()])

    # #### load test frames #####
    seq_dir = os.path.join(opt.dataset_dir, 'sequences', '%.2d' % opt.pose_test_seq)
    img_dir = os.path.join(seq_dir, 'image_2')

    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (opt.pose_test_seq, n) for n in range(N)]

    # #### load time file #####
    with open(opt.dataset_dir + '/sequences/%.2d/times.txt' % opt.pose_test_seq, 'r') as f:
        times = f.readlines()

    times = np.array([float(s[:-1]) for s in times])

    # #### Go! #####
    max_src_offset = (opt.seq_length - 1)
    all_pose = []
    pose_compare = []
    pred_all_list = []
    # ################
    # ## need argus ##
    # ################
    out_file = opt.output_dir + '/tre_result/'
    # one model file load
    out_file_model = out_file + opt.init_ckpt_file.split('/')[-1][6:] + '/'
    if not os.path.exists(out_file_model):
        os.makedirs(out_file_model)
    # txt file load
    out_txt_file = out_file_model + '{:02}'.format(opt.pose_test_seq) + '.txt'
    # summary result load
    result_sum = out_file + 'result_sum.txt'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        final_pose = None
        final = N - max_src_offset
        with open(out_txt_file, 'w') as f:
            for tgt_idx in range(0, N-max_src_offset, opt.batch_size):
                if (tgt_idx-max_src_offset) % 100 == 0:
                    print('Progress: %d/%d' % (tgt_idx-max_src_offset, N))

                inputs = np.zeros((opt.batch_size, opt.img_height, opt.img_width, 3*opt.seq_length), dtype=np.uint8)
                inputs_r = np.zeros((opt.batch_size, opt.img_height, opt.img_width, 3 * opt.seq_length), dtype=np.uint8)

                for b in range(opt.batch_size):
                    idx = tgt_idx + b
                    if idx >= N-max_src_offset:
                        break
                    image_seq = load_image_sequence(opt.dataset_dir,
                                                    test_frames,
                                                    idx,
                                                    opt.seq_length,
                                                    opt.img_height,
                                                    opt.img_width)
                    inputs[b] = image_seq

                    # add right images of tgt and src
                    image_seq_r = load_image_sequence_r(opt.dataset_dir,
                                                    test_frames,
                                                    idx,
                                                    opt.seq_length,
                                                    opt.img_height,
                                                    opt.img_width)
                    inputs_r[b] = image_seq_r

                # get K (intrinsics of camera)
                intrinsics_09 = np.array([239.926529, 0., 204.229309, 0., 244.615326, 63.346298, 0., 0., 1.]).reshape([1, 3, 3])

                pred = sess.run(fetches, feed_dict={input_uint8: inputs,
                                                    input_uint8_r: inputs_r,
                                                    intrinsics: intrinsics_09})

                pred_poses = pred['pose']

                # scale back to absolute scale
                for i in range(1, opt.num_source+1):
                    pred_poses[i][0][0:3, 3] = pred_poses[i][0][0:3, 3] * 0.3087

                # get 5frames reslut
                pred_all_list.append(pred_poses)
                temp_len = len(pred_all_list)
                if temp_len >= 4:
                    for b in range(opt.batch_size):
                        idx = tgt_idx + b
                        if idx >= N - max_src_offset:
                            break
                        pred_poses_0 = pred_all_list[temp_len - 4]  # [0,1-->0]
                        pred_poses_1 = np.matmul(pred_all_list[temp_len - 4][1], pred_all_list[temp_len - 3][1])  # [2-->0]
                        pred_poses_2 = np.matmul(pred_poses_1, pred_all_list[temp_len - 2][1])  # [3 --> 0]
                        pred_poses_3 = np.matmul(pred_poses_2, pred_all_list[temp_len - 1][1])  # [4-->0]
                        # pred_pose = pred_poses[b]
                        pred_poses_0.append(pred_poses_1)
                        pred_poses_0.append(pred_poses_2)
                        pred_poses_0.append(pred_poses_3)  # [0,1->0,2->0,3->0,4->0]  (5,1,4,4)
                        curr_times = times[idx - 2 - max_src_offset:idx - 2 + max_src_offset + 3]
                        # ate result for len 5
                        out_file_ate = opt.output_dir + '/ate_result/'
                        # one model file load
                        out_file_model_ate = out_file_ate + opt.init_ckpt_file.split('/')[-1][6:] + '/{:02}'.format(opt.pose_test_seq)
                        # txt file load
                        if not os.path.exists(out_file_model_ate):
                            os.makedirs(out_file_model_ate)
                        out_file = out_file_model_ate + '/%.6d.txt' % (idx - max_src_offset - 2)

                        dump_pose_seq_TUM(out_file, pred_poses_0, curr_times)

                # get tre
                for b in range(opt.batch_size):
                    pred_pose = pred_poses
                    pred_pose = [np.squeeze(pose) for pose in pred_pose]
                    pose_compare.append(pred_pose[opt.seq_length - 1])
                    if final_pose is None:
                        for i in range(2):
                            this_pose = pred_pose[i]
                            out_pose = np.reshape(this_pose, [-1])
                            out_pose = tuple([out_pose[i] for i in range(12)])
                            f.write('%f %f %f %f %f %f %f %f %f %f %f %f\n' % out_pose)
                            all_pose.append(this_pose)
                        final_pose = pred_pose[1]
                    else:
                        this_pose = np.dot(final_pose, pred_pose[1])
                        final_pose = np.dot(final_pose, pred_pose[1])
                        out_pose = np.reshape(this_pose, [-1])
                        out_pose = tuple([out_pose[i] for i in range(12)])
                        f.write('%f %f %f %f %f %f %f %f %f %f %f %f\n' % out_pose)
                        all_pose.append(this_pose)

    # eval 5frames ATE
    pred_dir = out_file_model_ate
    gtruth_dir = 'data/pose_gt/ate/{:02}/'.format(opt.pose_test_seq)
    test_model = opt.init_ckpt_file.split('/')[-1][6:]
    ate_summary_dir = out_file_ate
    eval_pose(pred_dir, gtruth_dir, test_model, ate_summary_dir)

    # eval tre (error of T and R)
    eval_ate = kittiEvalOdom('data/pose_gt/tre/', [opt.pose_test_seq])
    sequence = opt.pose_test_seq
    model_name = opt.init_ckpt_file.split('/')[-1]
    model_index = int(model_name.split('-')[-1])
    eval_ate.eval_sum(out_file_model, result_sum, model_index, sequence)

def load_image_sequence(dataset_dir,
                        frames,
                        tgt_idx,
                        seq_length,
                        img_height,
                        img_width):
    half_offset = int((seq_length - 1))
    for o in range(0, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        curr_img = sm.imread(img_file)
        curr_img = sm.imresize(curr_img, (img_height, img_width))
        if o == 0:
            image_seq = curr_img
        elif o == 1:
            image_seq = np.dstack((image_seq, curr_img))
        # else:
        #     image_seq = np.dstack((image_seq, curr_img))
    return image_seq

def load_image_sequence_r(dataset_dir,
                        frames,
                        tgt_idx,
                        seq_length,
                        img_height,
                        img_width):
    half_offset = int((seq_length - 1))
    for o in range(0, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_3/%s.png' % (curr_drive, curr_frame_id))
        curr_img = sm.imread(img_file)
        curr_img = sm.imresize(curr_img, (img_height, img_width))
        if o == 0:
            image_seq = curr_img
        elif o == 1:
            image_seq = np.dstack((image_seq, curr_img))
        # else:
        #     image_seq = np.dstack((image_seq, curr_img))
    return image_seq