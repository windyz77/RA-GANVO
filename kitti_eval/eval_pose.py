# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_pose.py
from __future__ import division
import os
import numpy as np
import argparse
from glob import glob
from kitti_eval.pose_evaluation_utils import *

def eval_pose(pred_dir,gtruth_dir, test_model,ate_summary_dir):
    pred_files = glob(pred_dir + '/*.txt')

    ate_all = []
    are_all = []
    for i in range(len(pred_files)):
        gtruth_file = gtruth_dir + os.path.basename(pred_files[i])
        if not os.path.exists(gtruth_file):
            continue
        are, ate = compute_ate(gtruth_file, pred_files[i])
        if ate == False:
            continue
        ate_all.append(ate)
        are_all.append(are)
    ate_all = np.array(ate_all)
    are_all = np.array(are_all)
    print("==============ATE RESULT FOR LEN {}==============".format(len(pred_files)))
    print("Predictions dir: %s" % pred_dir)
    print("ATE mean: %.4f, std: %.4f" % (np.mean(ate_all), np.std(ate_all)))
    print("ARE mean: %.4f, std: %.4f" % (np.mean(are_all), np.std(are_all)))
    with open(ate_summary_dir+'/ate_resultsum.txt', 'a') as f:
        f.write("==============model:{}==================\n".format(test_model))
        f.write("Predictions dir: %s\n" % pred_dir)
        f.write("ATE mean: %.4f, std: %.4f\n" % (np.mean(ate_all), np.std(ate_all)))
        f.write("ARE mean: %.4f, std: %.4f\n" % (np.mean(are_all), np.std(are_all)))
