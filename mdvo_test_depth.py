from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
from mdvo_model import *
# from kitti_eval.eval_depth import eval_depth

def test_depth(opt):
    # #### load testing list #####
    with open('data/kitti/test_files_%s.txt' % opt.depth_test_split, 'r') as f:
        test_files = f.readlines()
        test_files = [opt.dataset_dir + t[:-1] for t in test_files]

    with open('data/kitti/test_files_r_%s.txt' % opt.depth_test_split, 'r') as fr:
        test_files_r = fr.readlines()
        test_files_r = [opt.dataset_dir + t[:-1] for t in test_files_r]

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    # #### init #####
    input_tgt_uint8 = tf.placeholder(tf.uint8, [opt.batch_size, opt.img_height, opt.img_width, 3], name='raw_tgt_input')

    input_tgt_uint8_r = tf.placeholder(tf.uint8, [opt.batch_size, opt.img_height, opt.img_width, 3], name='raw_tgt_input_r')

    model = RA_GANVO_Model(opt, input_tgt_uint8, None, None, input_tgt_uint8_r, None)
    fetches = {"depth": model.pred_depth[0]}  # tgt depth

    saver = tf.train.Saver([var for var in tf.model_variables()])

    print(opt.output_dir + '/' + os.path.basename(opt.init_ckpt_file))
    # #### Go #####
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_option)) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        pred_all = []
        for t in range(0, len(test_files), opt.batch_size):
            if t % 100 == 0:
                print('processing: %d/%d' % (t, len(test_files)))
            inputs = np.zeros(
                (opt.batch_size, opt.img_height, opt.img_width, 3), dtype=np.uint8)

            inputs_r = np.zeros((opt.batch_size, opt.img_height, opt.img_width, 3), dtype=np.uint8)

            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                fh = open(test_files[idx], 'rb')
                fh_r = open(test_files_r[idx], 'rb')
                raw_im = pil.open(fh)
                raw_im_r = pil.open(fh_r)
                scaled_im = raw_im.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)
                scaled_im_r = raw_im_r.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)
                inputs[b] = np.array(scaled_im)
                inputs_r[b] = np.array(scaled_im_r)

            pred = sess.run(fetches, feed_dict={input_tgt_uint8: inputs, input_tgt_uint8_r: inputs_r})
            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred['depth'][b, :, :, 0])

        np.save(opt.output_dir + '/' + os.path.basename(opt.init_ckpt_file), pred_all)