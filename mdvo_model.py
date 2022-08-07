from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from optical_flow_warp_fwd import transformerFwd
from mdvo_nets import *
from utils import *
from optical_flow_warp_fwd import transformerFwd
from optical_flow_warp_old import transformer_old
from pwc_disp import construct_model_pwc_full_disp, feature_pyramid_disp, pwc_disp
from pwc_flow import feature_pyramid_flow, construct_model_pwc_full

class RA_GANVO_Model(object):

    def __init__(self, opt, tgt_image_left, src_image_left, intrinsics_left, tgt_image_right, src_right_image):
        self.opt = opt
        self.tgt_image = self.preprocess_image(tgt_image_left)
        self.src_image = self.preprocess_image(src_image_left)
        self.intrinsics = intrinsics_left
        self.tgt_image_right = self.preprocess_image(tgt_image_right)
        self.src_image_right = self.preprocess_image(src_right_image)

        self.build_model()
        if not opt.mode in ['train_rigid', 'train_flow']:
            return

        self.build_losses()
        self.bulid_sumarries()

    def build_model(self):
        opt = self.opt

        self.tgt_image_pyramid = self.scale_pyramid(self.tgt_image, opt.num_scales)
        self.src_image_pyramid = self.scale_pyramid(self.src_image, opt.num_scales)

        # add stereo ###
        self.tgt_image_pyramid_right = self.scale_pyramid(self.tgt_image_right, opt.num_scales)
        self.src_image_pyramid_right = self.scale_pyramid(self.src_image_right, opt.num_scales)

        if opt.add_dispnet:
            if opt.mode == 'train_rigid':
                self.build_flow_pwcnet()
                self.build_disp_pwcnet()
                self.build_flow_warping_pwcnet()
                self.bulid_disp_out()
            if opt.mode == 'test_depth':
                self.build_disp_pwcnet()

        # flow warppin
        if opt.add_posenet:
            if opt.mode == 'test_pose':
                self.build_flow_pwcnet()
                self.build_disp_pwcnet()

            self.build_TRTRTRTR_Refine_Net()

        if opt.add_dispnet and opt.add_posenet:
            self.build_TR_rigid_warping()   # tr
            self.build_2TR_rigid_warping()  # 2tr
            self.build_3TR_rigid_warping()  # 3tr
            self.build_4TR_rigid_warping()  # 4tr

        if opt.add_gan:
            if opt.mode == 'train_rigid':
                self.gan()

    # ########################################
    # ### use pwcnet get optical flow ########
    # ########################################
    def build_flow_pwcnet(self):
        opt = self.opt
        # save optical flow
        self.fwd_optical_flow = {}
        self.bwd_optical_flow = {}
        image1 = self.tgt_image    # (4,128,416,3)
        image2 = self.src_image

        with tf.variable_scope('train_flow', reuse=False):
            # get image feature
            self.feature_tgt = feature_pyramid_flow(image1, reuse=False)
            feature_src = feature_pyramid_flow(image2, reuse=True)
            # get optical flow
            fwd_optical_flow = construct_model_pwc_full(image1, image2, self.feature_tgt, feature_src, reuse=False)
            bwd_optical_flow = construct_model_pwc_full(image2, image1, feature_src, self.feature_tgt, reuse=True)
            self.fwd_optical_flow['TR'] = fwd_optical_flow    # 1 --> 2
            self.bwd_optical_flow['TR'] = bwd_optical_flow    # 2 --> 1

        self.pred_flow = [tf.concat([fwd_optical_flow[i], bwd_optical_flow[i]], axis=0) for i in range(opt.num_scales)]
        self.pred_flow_test = self.fwd_optical_flow['TR']   # use test optical flow

    def build_disp_pwcnet(self):
        opt = self.opt
        bs = opt.batch_size

        if opt.mode == 'test_depth':
            with tf.variable_scope('train_rigid_disp_feature_net', reuse=tf.AUTO_REUSE):
                self.tgt_feature = feature_pyramid_disp(self.tgt_image, reuse=tf.AUTO_REUSE)
                self.tgt_feature_r = feature_pyramid_disp(self.tgt_image_right, reuse=tf.AUTO_REUSE)

                self.pred_disp_tgt = pwc_disp(self.tgt_image, self.tgt_image_right, self.tgt_feature, self.tgt_feature_r,reuse=tf.AUTO_REUSE)

            self.pred_disp = [d[:bs, :, :, :] for d in self.pred_disp_tgt]  # only need disp of tgt
            # self.pred_depth = [1. / d for d in self.pred_disp]   # is depth
            self.pred_depth = [d for d in self.pred_disp]  # is disp, not depth
        else:
            with tf.variable_scope('train_rigid_disp_feature_net', reuse=False):
                self.tgt_feature = feature_pyramid_disp(self.tgt_image, reuse=False)
                self.tgt_feature_r = feature_pyramid_disp(self.tgt_image_right, reuse=True)
                self.src_feature = feature_pyramid_disp(self.src_image, reuse=True)
                self.src_feature_r = feature_pyramid_disp(self.src_image_right, reuse=True)

                self.pred_disp_tgt = pwc_disp(self.tgt_image, self.tgt_image_right, self.tgt_feature,
                                              self.tgt_feature_r, reuse=False)
                self.pred_disp_src = pwc_disp(self.src_image, self.src_image_right, self.src_feature,
                                              self.src_feature_r, reuse=True)

            self.pred_disp = [tf.concat([self.pred_disp_tgt[i][:bs, :, :, :], self.pred_disp_src[i][:bs, :, :, :]],
                              axis=0) for i in range(opt.num_scales)]

            self.pred_depth = [1. / d for d in self.pred_disp]

    def bulid_disp_out(self):
        opt = self.opt
        bs = opt.batch_size
        H = opt.img_height
        W = opt.img_width

        self.disp_tgt_est = [d[:bs, :, :, :] for d in self.pred_disp_tgt]
        self.disp_tgt_r_est = [d[bs:, :, :, :] for d in self.pred_disp_tgt]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_tgt_smoothness = self.get_disparity_smoothness_2nd(self.disp_tgt_est, self.tgt_image_pyramid)
            self.disp_tgt_r_smoothness = self.get_disparity_smoothness_2nd(self.disp_tgt_r_est,
                                                                           self.tgt_image_pyramid_right)
            # lr consice
            self.tgt_ltr_flow = [self.generate_flow_left(self.disp_tgt_est[i], i) for i in range(4)]
            self.tgt_rtl_flow = [self.generate_flow_right(self.disp_tgt_r_est[i], i) for i in range(4)]

        self.tgt_right_occ_mask = [
            tf.clip_by_value(
                transformerFwd(
                    tf.ones(
                        shape=[bs, H // (2 ** i), W // (2 ** i), 1],
                        dtype='float32'),
                    self.tgt_ltr_flow[i], [H // (2 ** i), W // (2 ** i)]),
                clip_value_min=0.0,
                clip_value_max=1.0) for i in range(4)]

        self.tgt_left_occ_mask = [
            tf.clip_by_value(
                transformerFwd(
                    tf.ones(
                        shape=[
                            bs, H // (2 ** i), W // (2 ** i), 1
                        ],
                        dtype='float32'),
                    self.tgt_rtl_flow[i], [H // (2 ** i), W // (2 ** i)]),
                clip_value_min=0.0,
                clip_value_max=1.0) for i in range(4)]

        self.tgt_right_occ_mask_avg = [tf.reduce_mean(self.tgt_right_occ_mask[i]) + 1e-12 for i in range(4)]
        self.tgt_left_occ_mask_avg = [tf.reduce_mean(self.tgt_left_occ_mask[i]) + 1e-12 for i in range(4)]

        # GENERATE IMAGES
        with tf.variable_scope('images'):
            self.tgt_left_est = [self.generate_transformed(self.tgt_image_pyramid_right[i], self.tgt_ltr_flow[i], i)
                                 for i in range(4)]

            self.tgt_right_est = [self.generate_transformed(self.tgt_image_pyramid[i], self.tgt_rtl_flow[i], i)
                                  for i in range(4)]
        # LR CONSISTENCY
        with tf.variable_scope('left-right'):
            self.tgt_right_to_left_disp = [self.generate_transformed(self.disp_tgt_r_est[i], self.tgt_ltr_flow[i], i)
                                           for i in range(4)]

            self.tgt_left_to_right_disp = [self.generate_transformed(self.disp_tgt_est[i], self.tgt_rtl_flow[i], i) for
                                           i in range(4)]

    # #########################################
    # ## refine Decoupled Pose-Net (TRTRTRTR)##
    # #########################################
    def build_TRTRTRTR_Refine_Net(self):
        opt = self.opt
        bs = opt.batch_size
        self.pose = {}
        fwd_optical_flow = self.fwd_optical_flow['TR']  # t --> s optical flow

        # Input
        Input = tf.concat([self.tgt_image, self.src_image, fwd_optical_flow[0][:bs]], axis=3)  # (4,128,416,8)
        # #########
        # # T-NET #
        # #########
        refine_T = T_Refine_Net(opt, Input)
        # convert to matrix
        T_matrix = []
        for i in range(opt.num_source):
            T_matrix.append(pose_vec2mat(refine_T[:, i, :]))
        # #########
        # # R-NET #
        # #########
        Input = self.generate_feature_images(T_matrix, fwd_optical_flow[0][:bs])
        # refine R
        refine_R = R_Refine_Net(opt, Input)

        # convert to matrix
        refine_R_matrix = []
        for i in range(opt.num_source):
            refine_R_matrix.append(pose_vec2mat(refine_R[:, i, :]))
        # refine pose matrix
        self.TR_matrix = []
        for i in range(opt.num_source):
            self.TR_matrix.append(tf.matmul(T_matrix[i], refine_R_matrix[i]))

        self.pose['TR'] = self.TR_matrix[0]
        # ##########
        # # TT-NET #
        # ##########
        Input, fwd_optical_flow, _ = self.generate_feature_images_2(self.TR_matrix)
        # refine T
        refine_TT = TT_Refine_Net(opt, Input)
        # convert to matrix
        refine_TT_matrix = []
        for i in range(opt.num_source):
            refine_TT_matrix.append(pose_vec2mat(refine_TT[:, i, :]))
        # refine pose matrix
        TRT_matrix = []
        for i in range(opt.num_source):
            TRT_matrix.append(tf.matmul(refine_TT_matrix[i], self.TR_matrix[i]))
        # ##########
        # # RR-NET #
        # ##########
        Input = self.generate_feature_images(TRT_matrix, fwd_optical_flow[0][:bs])
        # refine R
        refine_RR = RR_Refine_Net(opt, Input)
        # convert to matrix
        refine_RR_matrix = []
        for i in range(opt.num_source):
            refine_RR_matrix.append(pose_vec2mat(refine_RR[:, i, :]))
        # refine pose matrix
        self.TRTR_matrix = []
        for i in range(opt.num_source):
            self.TRTR_matrix.append(tf.matmul(TRT_matrix[i], refine_RR_matrix[i]))

        self.pose['2TR'] = self.TRTR_matrix[0]

        # ###########
        # # TTT-NET #
        # ###########
        Input, fwd_optical_flow, _ = self.generate_feature_images_2(self.TRTR_matrix)
        # refine T
        refine_TTT = TTT_Refine_Net(opt,Input)
        # convert to matrix
        refine_TTT_matrix = []
        for i in range(opt.num_source):
            refine_TTT_matrix.append(pose_vec2mat(refine_TTT[:, i, :]))
        # refine pose matrix
        TRTRT_matrix = []
        for i in range(opt.num_source):
            TRTRT_matrix.append(tf.matmul(refine_TTT_matrix[i], self.TRTR_matrix[i]))
        # # ###########
        # # # RRR-NET #
        # # ###########
        Input = self.generate_feature_images(TRTRT_matrix, fwd_optical_flow[0][:bs])
        # refine R
        refine_RRR = RRR_Refine_Net(opt, Input)
        # convert to matrix
        refine_RRR_matrix = []
        for i in range(opt.num_source):
            refine_RRR_matrix.append(pose_vec2mat(refine_RRR[:, i, :]))
        # refine pose matrix
        self.TRTRTR_matrix = []
        for i in range(opt.num_source):
            self.TRTRTR_matrix.append(tf.matmul(TRTRT_matrix[i], refine_RRR_matrix[i]))

        self.pose['3TR'] = self.TRTRTR_matrix[0]

        # # ############
        # # # TTTT-NET #
        # # ############
        Input, fwd_optical_flow, _ = self.generate_feature_images_2(self.TRTRTR_matrix)
        # refine T
        refine_TTTT = TTTT_Refine_Net(opt, Input)
        # convert to matrix
        refine_TTTT_matrix = []
        for i in range(opt.num_source):
            refine_TTTT_matrix.append(pose_vec2mat(refine_TTTT[:, i, :]))
        # refine pose matrix
        TRTRTRT_matrix = []
        for i in range(opt.num_source):
            TRTRTRT_matrix.append(tf.matmul(refine_TTTT_matrix[i], self.TRTRTR_matrix[i]))
        # # ############
        # # # RRRR-NET #
        # # ############
        Input = self.generate_feature_images(TRTRTRT_matrix, fwd_optical_flow[0][:bs])
        # refine R
        refine_RRRR = RRRR_Refine_Net(opt, Input)

        # convert to matrix
        refine_RRRR_matrix = []
        for i in range(opt.num_source):
            refine_RRRR_matrix.append(pose_vec2mat(refine_RRRR[:, i, :]))

        # refine pose matrix
        self.TRTRTRTR_matrix = []
        for i in range(opt.num_source):
            self.TRTRTRTR_matrix.append(tf.matmul(TRTRTRT_matrix[i], refine_RRRR_matrix[i]))

        self.pose['4TR'] = self.TRTRTRTR_matrix[0]
        _, self.fwd_optical_flow['5TR'], self.bwd_optical_flow['5TR'] = self.generate_feature_images_2(self.TRTRTRTR_matrix)

        # 2 frames pose test
        if opt.mode == 'test_pose':
            first_pose = tf.eye(4, batch_shape=[opt.batch_size])
            matrix0 = tf.matrix_inverse(self.TRTRTRTR_matrix[0])  # s --> t
            self.pred_rt_full = [first_pose, matrix0]

    def build_flow_warping_pwcnet(self):
        opt = self.opt
        image1 = self.tgt_image
        image2 = self.src_image
        batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])
        self.summaries = []

        self.occu_masks = [
            tf.clip_by_value(
                transformerFwd(
                    tf.ones(
                        shape=[batch_size, H / (2 ** s), W / (2 ** s), 1],
                        dtype='float32'),
                    flowr, [H / (2 ** s), W / (2 ** s)]),
                clip_value_min=0.0,
                clip_value_max=1.0)
            for s, flowr in enumerate(self.bwd_optical_flow['TR'])]

        self.pixel_loss_depth = 0
        self.pixel_loss_optical = 0
        self.exp_loss = 0
        self.flow_smooth_loss = 0
        self.tgt_image_all = []
        self.src_image_all = []
        self.proj_image_depth_all = []
        self.proj_error_depth_all = []
        self.flyout_map_all = []

        for s in range(opt.num_scales):
            # Scale the source and target images for computing loss at the
            # according scale.
            curr_tgt_image = tf.image.resize_area(
                image1,
                [int(opt.img_height / (2 ** s)), int(opt.img_width / (2 ** s))])
            curr_src_image = tf.image.resize_area(
                image2,
                [int(opt.img_height / (2 ** s)), int(opt.img_width / (2 ** s))])

            occu_mask = self.occu_masks[s]
            occu_mask_avg = tf.reduce_mean(occu_mask)

            curr_proj_image_optical = transformer_old(
                curr_src_image, self.fwd_optical_flow['TR'][s], [H / (2 ** s), W / (2 ** s)])

            curr_proj_error_optical = tf.abs(curr_proj_image_optical -
                                             curr_tgt_image)
            self.pixel_loss_optical += (1.0 - opt.ssim_weight) * tf.reduce_mean(
                curr_proj_error_optical * occu_mask) / occu_mask_avg

            if opt.ssim_weight > 0:
                self.pixel_loss_optical += opt.ssim_weight * tf.reduce_mean(
                    self.unos_SSIM(curr_proj_image_optical * occu_mask, curr_tgt_image *occu_mask)) / occu_mask_avg

            self.flow_smooth_loss += opt.flow_smooth_weight * self.unos_cal_grad2_error(
                    self.fwd_optical_flow['TR'][s]/20.0, curr_tgt_image, 1.0)

        self.optical_flow_loss = (self.pixel_loss_optical + self.flow_smooth_loss)

    def build_TR_rigid_warping(self):
        opt = self.opt
        bs = opt.batch_size

        self.fwd_rigid_warp_pyramid_TR = []
        self.bwd_rigid_warp_pyramid_TR = []

        # build rigid flow (fwd: tgt->src, bwd: src->tgt)
        # # opt.num_source = seq - 1 =1
        # # opt.num_scales = 4
        for s in range(opt.num_scales):
            i = 0
            fwd_rigid_flow = compute_rigid_flow_quater_change(tf.squeeze(self.pred_depth[s][:bs], axis=3),
                                                              self.pose['TR'],
                                                              self.intrinsics[:, s, :, :],
                                                              False)
            bwd_rigid_flow = compute_rigid_flow_quater_change(
                    tf.squeeze(self.pred_depth[s][bs * (i + 1):bs * (i + 2)], axis=3),
                    self.pose['TR'],
                    self.intrinsics[:, s, :, :],
                    True)
            # warping by rigid flow
            self.fwd_rigid_warp_pyramid_TR.append(flow_warp(self.src_image_pyramid[s], fwd_rigid_flow))
            self.bwd_rigid_warp_pyramid_TR.append(flow_warp(self.tgt_image_pyramid[s], bwd_rigid_flow))

    def build_2TR_rigid_warping(self):
        opt = self.opt
        bs = opt.batch_size

        self.fwd_rigid_warp_pyramid_2TR = []
        self.bwd_rigid_warp_pyramid_2TR = []

        for s in range(opt.num_scales):
            i = 0
            fwd_rigid_flow = compute_rigid_flow_quater_change(tf.squeeze(self.pred_depth[s][:bs], axis=3),
                                                                      self.pose['2TR'],
                                                                      self.intrinsics[:, s, :, :],
                                                                      False)
            bwd_rigid_flow = compute_rigid_flow_quater_change(
                        tf.squeeze(self.pred_depth[s][bs * (i + 1):bs * (i + 2)], axis=3),
                        self.pose['2TR'],
                        self.intrinsics[:, s, :, :],
                        True)

            # warping by rigid flow
            self.fwd_rigid_warp_pyramid_2TR.append(flow_warp(self.src_image_pyramid[s], fwd_rigid_flow))

            self.bwd_rigid_warp_pyramid_2TR.append(flow_warp(self.tgt_image_pyramid[s], bwd_rigid_flow))

    def build_3TR_rigid_warping(self):
        opt = self.opt
        bs = opt.batch_size

        self.fwd_rigid_warp_pyramid_3TR = []
        self.bwd_rigid_warp_pyramid_3TR = []

        for s in range(opt.num_scales):
            i = 0
            fwd_rigid_flow = compute_rigid_flow_quater_change(
                tf.squeeze(self.pred_depth[s][:bs], axis=3),
                self.pose['3TR'],
                self.intrinsics[:, s, :, :],
                False)
            bwd_rigid_flow = compute_rigid_flow_quater_change(
                tf.squeeze(self.pred_depth[s][bs * (i + 1):bs * (i + 2)], axis=3),
                self.pose['3TR'], self.intrinsics[:, s, :, :],
                True)

            # warping by rigid flow
            self.fwd_rigid_warp_pyramid_3TR.append(flow_warp(self.src_image_pyramid[s], fwd_rigid_flow))
            self.bwd_rigid_warp_pyramid_3TR.append(flow_warp(self.tgt_image_pyramid[s], bwd_rigid_flow))


    def build_4TR_rigid_warping(self):
        opt = self.opt
        bs = opt.batch_size
        H = opt.img_height
        W = opt.img_width

        self.fwd_ref_exp_masks_4TR = []
        self.bwd_ref_exp_masks_4TR = []

        self.fwd_rigid_warp_pyramid_4TR = []
        self.bwd_rigid_warp_pyramid_4TR = []
        self.fwd_rigid_flow = []
        self.bwd_rigid_flow = []


        self.flow_consist_loss = 0
        self.fwd_flow_consist_loss = 0
        self.bwd_flow_consist_loss = 0

        # add mask
        # fwd_mask
        self.fwd_occu_masks_4TR = [
            tf.clip_by_value(
                transformerFwd(
                    tf.ones(
                        shape=[bs, H / (2 ** s), W / (2 ** s), 1],
                        dtype='float32'),
                    flowr, [H / (2 ** s), W / (2 ** s)]),
                clip_value_min=0.0,
                clip_value_max=1.0)
            for s, flowr in enumerate(self.bwd_optical_flow['5TR'])]
        self.bwd_occu_masks_4TR = [
            tf.clip_by_value(
                transformerFwd(
                    tf.ones(
                        shape=[bs, H / (2 ** s), W / (2 ** s), 1],
                        dtype='float32'),
                    flowr, [H / (2 ** s), W / (2 ** s)]),
                clip_value_min=0.0,
                clip_value_max=1.0)
            for s, flowr in enumerate(self.fwd_optical_flow['5TR'])]

        for s in range(opt.num_scales):
            i = 0

            fwd_rigid_flow = compute_rigid_flow_quater_change(
                tf.squeeze(self.pred_depth[s][:bs], axis=3),
                self.pose['4TR'],
                self.intrinsics[:, s, :, :],
                False)
            bwd_rigid_flow = compute_rigid_flow_quater_change(
                tf.squeeze(self.pred_depth[s][bs * (i + 1):bs * (i + 2)], axis=3),
                self.pose['4TR'], self.intrinsics[:, s, :, :],
                True)
            self.fwd_rigid_flow.append(fwd_rigid_flow)
            self.bwd_rigid_flow.append(bwd_rigid_flow)
            #########################################
            # add fwd diff mask
            fwd_occu_mask = self.fwd_occu_masks_4TR[s]
            fwd_flow_diff = tf.sqrt(
                tf.reduce_sum(
                    tf.square(fwd_rigid_flow - self.fwd_optical_flow['5TR'][s]),
                    axis=3,
                    keep_dims=True))
            fwd_flow_diff_mask = tf.cast(
                fwd_flow_diff < (opt.flow_diff_threshold / 2 ** s), tf.float32)
            fwd_occu_region = tf.cast(fwd_occu_mask < 0.5, tf.float32)
            fwd_ref_exp_mask = tf.clip_by_value(
                fwd_flow_diff_mask + fwd_occu_region,
                clip_value_min=0.0,
                clip_value_max=1.0)
            self.fwd_ref_exp_masks_4TR.append(fwd_ref_exp_mask)

            # add bwd diff_mask
            bwd_occu_mask = self.bwd_occu_masks_4TR[s]
            bwd_flow_diff = tf.sqrt(
                tf.reduce_sum(
                    tf.square(bwd_rigid_flow - self.bwd_optical_flow["5TR"][s]),
                    axis=3,
                    keep_dims=True))
            bwd_flow_diff_mask = tf.cast(
                bwd_flow_diff < (opt.flow_diff_threshold / 2 ** s), tf.float32)
            bwd_occu_region = tf.cast(bwd_occu_mask < 0.5, tf.float32)
            bwd_ref_exp_mask = tf.clip_by_value(
                bwd_flow_diff_mask + bwd_occu_region,
                clip_value_min=0.0,
                clip_value_max=1.0)
            self.bwd_ref_exp_masks_4TR.append(bwd_ref_exp_mask)

            # warping by rigid flow
            self.fwd_rigid_warp_pyramid_4TR.append(flow_warp(self.src_image_pyramid[s], fwd_rigid_flow))
            self.bwd_rigid_warp_pyramid_4TR.append(flow_warp(self.tgt_image_pyramid[s], bwd_rigid_flow))

    def gan(self):
        opt = self.opt
        # add mask gan
        self.fwd_image_4TR = self.fwd_rigid_warp_pyramid_4TR[0] * self.fwd_ref_exp_masks_4TR[0] * self.fwd_occu_masks_4TR[0]
        self.bwd_image_4TR = self.bwd_rigid_warp_pyramid_4TR[0] * self.bwd_ref_exp_masks_4TR[0] * self.bwd_occu_masks_4TR[0]

        # real image
        self.input_tgt_4TR = self.tgt_image * self.fwd_ref_exp_masks_4TR[0] * self.fwd_occu_masks_4TR[0]
        self.input_src_4TR = self.src_image * self.bwd_ref_exp_masks_4TR[0] * self.bwd_occu_masks_4TR[0]

        # GAN D-net .
        self.discriminate_real_fwd_4tr = discriminate_net_2(opt, self.input_tgt_4TR, self.input_tgt_4TR, reuse=False)
        self.discriminate_fake_fwd_4tr = discriminate_net_2(opt, self.input_tgt_4TR, self.fwd_image_4TR, reuse=True)

        self.discriminate_real_bwd_4tr = discriminate_net_2(opt, self.input_src_4TR, self.input_src_4TR, reuse=True)
        self.discriminate_fake_bwd_4tr = discriminate_net_2(opt, self.input_src_4TR, self.bwd_image_4TR, reuse=True)

    def build_losses(self):
        opt = self.opt
        self.disp_smooth_loss = 0

        self.fwd_rigid_error_pyramid_TR = 0
        self.bwd_rigid_error_pyramid_TR = 0
        self.fwd_rigid_error_pyramid_2TR = 0
        self.bwd_rigid_error_pyramid_2TR = 0
        self.fwd_rigid_error_pyramid_3TR = 0
        self.bwd_rigid_error_pyramid_3TR = 0
        self.fwd_rigid_error_pyramid_4TR = 0
        self.bwd_rigid_error_pyramid_4TR = 0

        if opt.add_gan:
            EPS = 1e-12
            self.d_loss = 0
            self.g_loss = 0
            # ##############
            # ###gan loss###
            # ##############
            ###  d-net loss
            self.d_loss += 0.0001 * tf.reduce_mean(-(tf.log(self.discriminate_real_fwd_4tr + EPS) +
                                                    tf.log(self.discriminate_real_bwd_4tr + EPS) +
                                                    tf.log(1 - self.discriminate_fake_fwd_4tr + EPS) +
                                                    tf.log(1 - self.discriminate_fake_bwd_4tr + EPS)))

            self.g_loss += 0.0001 * tf.reduce_mean(
                -(tf.log(self.discriminate_fake_fwd_4tr + EPS) + tf.log(self.discriminate_fake_bwd_4tr + EPS)))

        if opt.mode == 'train_rigid':
            ###########################################################################################
            # add pose loss
            # compute reconstruction error
            for s in range(opt.num_scales):
                # TR
                self.fwd_rigid_error_pyramid_TR += 0.2 * opt.rigid_warp_weight * opt.num_source / 2 * \
                                                   tf.reduce_mean(self.image_similarity_add_mask(
                                                       self.fwd_rigid_warp_pyramid_TR[s],
                                                       self.tgt_image_pyramid[s],
                                                       self.fwd_occu_masks_4TR[s],
                                                       self.fwd_ref_exp_masks_4TR[s]))
                self.bwd_rigid_error_pyramid_TR += 0.2 * opt.rigid_warp_weight * opt.num_source / 2 * \
                                                   tf.reduce_mean(self.image_similarity_add_mask(
                                                       self.bwd_rigid_warp_pyramid_TR[s],
                                                       self.src_image_pyramid[s],
                                                       self.bwd_occu_masks_4TR[s],
                                                       self.bwd_ref_exp_masks_4TR[s]))
                # 2TR
                # compute reconstruction error
                self.fwd_rigid_error_pyramid_2TR += opt.rigid_warp_weight * opt.num_source / 2 * \
                                                    tf.reduce_mean(self.image_similarity_add_mask(
                                                        self.fwd_rigid_warp_pyramid_2TR[s],
                                                        self.tgt_image_pyramid[s],
                                                        self.fwd_occu_masks_4TR[s],
                                                        self.fwd_ref_exp_masks_4TR[s]))

                self.bwd_rigid_error_pyramid_2TR += opt.rigid_warp_weight * opt.num_source / 2 * \
                                                    tf.reduce_mean(self.image_similarity_add_mask(
                                                        self.bwd_rigid_warp_pyramid_2TR[s],
                                                        self.src_image_pyramid[s],
                                                        self.bwd_occu_masks_4TR[s],
                                                        self.bwd_ref_exp_masks_4TR[s]))

                # 3TR
                # compute reconstruction error
                self.fwd_rigid_error_pyramid_3TR += opt.rigid_warp_weight * opt.num_source / 2 * \
                                                     tf.reduce_mean(self.image_similarity_add_mask(
                                                        self.fwd_rigid_warp_pyramid_3TR[s],
                                                        self.tgt_image_pyramid[s],
                                                        self.fwd_occu_masks_4TR[s],
                                                        self.fwd_ref_exp_masks_4TR[s]))

                self.bwd_rigid_error_pyramid_3TR += opt.rigid_warp_weight * opt.num_source / 2 * \
                                                     tf.reduce_mean(self.image_similarity_add_mask(
                                                        self.bwd_rigid_warp_pyramid_3TR[s],
                                                        self.src_image_pyramid[s],
                                                        self.bwd_occu_masks_4TR[s],
                                                        self.bwd_ref_exp_masks_4TR[s]))
                # 4TR
                # compute reconstruction error
                self.fwd_rigid_error_pyramid_4TR += opt.rigid_warp_weight * opt.num_source / 2 * \
                                                    tf.reduce_mean(self.image_similarity(
                                                        self.fwd_rigid_warp_pyramid_4TR[s],
                                                        self.tgt_image_pyramid[s]))
                self.bwd_rigid_error_pyramid_4TR += opt.rigid_warp_weight * opt.num_source / 2 * \
                                                    tf.reduce_mean(self.image_similarity(
                                                        self.bwd_rigid_warp_pyramid_4TR[s],
                                                        self.src_image_pyramid[s]))

            self.rigid_warp_loss_TR = (self.fwd_rigid_error_pyramid_TR + self.bwd_rigid_error_pyramid_TR)
            self.rigid_warp_loss_2TR = (self.fwd_rigid_error_pyramid_2TR + self.bwd_rigid_error_pyramid_2TR)
            self.rigid_warp_loss_3TR = (self.fwd_rigid_error_pyramid_3TR + self.bwd_rigid_error_pyramid_3TR)
            self.rigid_warp_loss_4TR = (self.fwd_rigid_error_pyramid_4TR + self.bwd_rigid_error_pyramid_4TR)

            # pwc disp loss
            with tf.variable_scope('losses', reuse=None):
                # IMAGE RECONSTRUCTION
                # L1
                self.tgt_l1_left = [tf.abs(self.tgt_left_est[i] - self.tgt_image_pyramid[i]) * self.tgt_left_occ_mask[i]
                                    for i in range(4)]
                self.tgt_l1_reconstruction_loss_left = [tf.reduce_mean(l) / self.tgt_left_occ_mask_avg[i]
                                                        for i, l in enumerate(self.tgt_l1_left)]

                self.tgt_l1_right = [tf.abs(self.tgt_right_est[i] - self.tgt_image_pyramid_right[i]) * self.tgt_right_occ_mask[i]
                                     for i in range(4)]
                self.tgt_l1_reconstruction_loss_right = [tf.reduce_mean(l) / self.tgt_right_occ_mask_avg[i]
                                                         for i, l in enumerate(self.tgt_l1_right)]

                # SSIM
                self.tgt_ssim_left = [self.unos_SSIM(self.tgt_left_est[i] * self.tgt_left_occ_mask[i],
                                      self.tgt_image_pyramid[i] * self.tgt_left_occ_mask[i])
                                      for i in range(4)]
                self.tgt_ssim_loss_left = [tf.reduce_mean(s) / self.tgt_left_occ_mask_avg[i]
                                           for i, s in enumerate(self.tgt_ssim_left)]

                self.tgt_ssim_right = [self.unos_SSIM(self.tgt_right_est[i] * self.tgt_right_occ_mask[i],
                                       self.tgt_image_pyramid_right[i] * self.tgt_right_occ_mask[i])
                                       for i in range(4)]
                self.tgt_ssim_loss_right = [tf.reduce_mean(s) / self.tgt_right_occ_mask_avg[i]
                                            for i, s in enumerate(self.tgt_ssim_right)]

                # WEIGTHED SUM
                self.tgt_image_loss_right = [self.opt.image_loss * self.tgt_ssim_loss_right[i] +
                                             (1 - self.opt.image_loss) * self.tgt_l1_reconstruction_loss_right[i]
                                             for i in range(4)]

                self.tgt_image_loss_left = [self.opt.image_loss * self.tgt_ssim_loss_left[i] +
                                            (1 - self.opt.image_loss) * self.tgt_l1_reconstruction_loss_left[i]
                                            for i in range(4)]

                self.image_loss_tgt = tf.add_n(self.tgt_image_loss_left + self.tgt_image_loss_right)

                self.image_loss = self.image_loss_tgt

            # DISPARITY SMOOTHNESS
            self.disp_tgt_loss = [tf.reduce_mean(tf.abs(self.disp_tgt_smoothness[i])) / 2 ** i for i in range(4 * 2)]
            self.disp_tgt_r_loss = [tf.reduce_mean(tf.abs(self.disp_tgt_r_smoothness[i])) / 2 ** i for i in range(4 * 2)]

            self.disp_lr_smooth = tf.add_n(self.disp_tgt_loss + self.disp_tgt_r_loss) * 0.5

            # disp lr consistency loss
            self.tgt_lr_left_loss = [tf.reduce_mean(tf.abs(self.tgt_right_to_left_disp[i] - self.disp_tgt_est[i]) *
                                                    self.tgt_left_occ_mask[i]) for i in range(4)]
            self.tgt_lr_right_loss = [tf.reduce_mean(tf.abs(self.tgt_left_to_right_disp[i] - self.disp_tgt_r_est[i]) *
                                                     self.tgt_right_occ_mask[i]) for i in range(4)]

            self.tgt_lr_loss = tf.add_n(self.tgt_lr_left_loss + self.tgt_lr_right_loss)

            self.lr_loss = self.tgt_lr_loss
            # depth total loss
            self.depth_loss = 10*self.disp_lr_smooth + self.image_loss + self.lr_loss

        self.rigid_loss = 0
        self.d_loss_finall = 0
        self.g_loss_finall = 0

        if opt.mode == 'train_rigid':
            self.rigid_loss += self.rigid_warp_loss_TR + self.rigid_warp_loss_2TR + \
                               self.rigid_warp_loss_3TR + self.rigid_warp_loss_4TR + \
                               self.depth_loss + self.optical_flow_loss
        if opt.add_gan:
           self.d_loss_finall += self.d_loss
           self.g_loss_finall += self.rigid_loss + self.g_loss


    # tensor board
    def bulid_sumarries(self):
        # tensorbaord 可视化
        n = self.opt.batch_size
        if self.opt.mode == 'train_rigid':
            # show images
            tgt_disp = self.pred_disp[0][:n, :, :, :]
            fwd_warp = self.fwd_rigid_warp_pyramid_4TR[0][:n, :, :, :]
            bwd_warp = self.bwd_rigid_warp_pyramid_4TR[0][:n, :, :, :]

            fwd_occu_mask_4TR = self.fwd_occu_masks_4TR[0][:n, :, :, :]
            bwd_occu_mask_4TR = self.bwd_occu_masks_4TR[0][:n, :, :, :]
            fwd_ref_mask_4TR = self.fwd_ref_exp_masks_4TR[0][:n, :, :, :]
            bwd_ref_mask_4TR = self.bwd_ref_exp_masks_4TR[0][:n, :, :, :]

            # add loss summar
            tf.summary.scalar('depth total loss:', self.depth_loss, collections=['summary_see'])
            tf.summary.scalar('total_loss:', self.rigid_loss, collections=['summary_see'])
            tf.summary.scalar('optical_flow_total_loss:', self.optical_flow_loss, collections=['summary_see'])
            tf.summary.scalar('flow_smooth_loss:', self.flow_smooth_loss, collections=['summary_see'])
            tf.summary.scalar('flow_pixe_loss', self.pixel_loss_optical, collections=['summary_see'])
            tf.summary.scalar('TR rigid TR loss:', self.rigid_warp_loss_TR, collections=['summary_see'])
            tf.summary.scalar('2TR rigid 2TRloss', self.rigid_warp_loss_2TR, collections=['summary_see'])
            tf.summary.scalar('3TR rigid 3TRloss', self.rigid_warp_loss_3TR, collections=['summary_see'])
            tf.summary.scalar('4TR rigid 4TRloss', self.rigid_warp_loss_4TR, collections=['summary_see'])
            tf.summary.scalar('Pwc-disp smooth loss:', self.disp_lr_smooth, collections=['summary_see'])
            tf.summary.scalar('Pwc-disp lr consisitency loss:', self.lr_loss, collections=['summary_see'])
            tf.summary.scalar('d_loss:', self.d_loss_finall, collections=['summary_see'])
            tf.summary.scalar('g_loss:', self.g_loss, collections=['summary_see'])
            tf.summary.scalar('g_loss_finall:', self.g_loss_finall, collections=['summary_see'])

            # add image summary
            tf.summary.image('center img', self.tgt_image * 255, max_outputs=self.opt.batch_size, collections=['summary_see'])
            tf.summary.image('tgt_disp', tgt_disp, max_outputs=self.opt.batch_size, collections=['summary_see'])

            tf.summary.image('fwd_warp_4TR', fwd_warp, max_outputs=self.opt.batch_size, collections=['summary_see'])
            tf.summary.image('bwd_warp_4TR', bwd_warp, max_outputs=self.opt.batch_size, collections=['summary_see'])

            # add mask summary
            tf.summary.image('fwd_occu_mask_4TR', fwd_occu_mask_4TR, max_outputs=self.opt.batch_size, collections=['summary_see'])
            tf.summary.image('bwd_occu_mask_4TR', bwd_occu_mask_4TR, max_outputs=self.opt.batch_size, collections=['summary_see'])
            tf.summary.image('fwd_ref_exp_mask_4TR', fwd_ref_mask_4TR, max_outputs=self.opt.batch_size, collections=['summary_see'])
            tf.summary.image('bwd_ref_exp_mask_4TR', bwd_ref_mask_4TR, max_outputs=self.opt.batch_size, collections=['summary_see'])


    def generate_flow_right(self, disp, scale):
        return self.generate_flow_left(-disp, scale)

    def generate_flow_left(self, disp, scale):
        batch_size = self.opt.batch_size
        H = self.opt.img_height // (2**scale)
        W = self.opt.img_width // (2**scale)

        zero_flow = tf.zeros([batch_size, H, W, 1])
        ltr_flow = -disp * W
        ltr_flow = tf.concat([ltr_flow, zero_flow], axis=3)
        return ltr_flow

    def generate_transformed(self, img, flow, scale):
        return transformer_old(
            img,
            flow,
            out_size=[
                self.opt.img_height // (2**scale),
                self.opt.img_width // (2**scale)
            ])

    def get_disparity_smoothness_2nd(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        disp_gradients_xx = [self.gradient_x(dg) for dg in disp_gradients_x]
        disp_gradients_yy = [self.gradient_y(dg) for dg in disp_gradients_y]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [
            tf.exp(-tf.reduce_mean(
                10.0 * tf.abs(g), 3, keep_dims=True))
            for g in image_gradients_x
        ]
        weights_y = [
            tf.exp(-tf.reduce_mean(
                10.0 * tf.abs(g), 3, keep_dims=True))
            for g in image_gradients_y
        ]

        smoothness_x = [
            disp_gradients_xx[i] * weights_x[i][:, :, :-1, :] for i in range(4)
        ]
        smoothness_y = [
            disp_gradients_yy[i] * weights_y[i][:, :-1, :, :] for i in range(4)
        ]
        return smoothness_x + smoothness_y


    def unos_cal_grad2_error(self, flo, image, beta):
        """
        Calculate the image-edge-aware second-order smoothness loss for flo
        """

        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy

        img_grad_x, img_grad_y = gradient(image)
        weights_x = tf.exp(-10.0 * tf.reduce_mean(
            tf.abs(img_grad_x), 3, keep_dims=True))
        weights_y = tf.exp(-10.0 * tf.reduce_mean(
            tf.abs(img_grad_y), 3, keep_dims=True))

        dx, dy = gradient(flo)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)

        return (tf.reduce_mean(beta * weights_x[:, :, 1:, :] * tf.abs(dx2))+tf.reduce_mean(beta * weights_y[:, 1:, :, :] * tf.abs(dy2))) / 2.0

    def unos_SSIM(self, x, y):

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    # extra functions
    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def image_similarity_add_mask(self, x, y, occu_mask, ref_mask):
        occu_mask_avg = tf.reduce_mean(occu_mask)
        # 先 L1 loss
        L1_loss = tf.abs(x - y) * ref_mask
        pixel_loss = (1.0 - self.opt.ssim_weight) * tf.reduce_mean(L1_loss * occu_mask) / occu_mask_avg
        # 后 ssim loss
        ssim_loss = self.opt.ssim_weight * tf.reduce_mean(self.unos_SSIM(x * occu_mask * ref_mask,
                                                                         y * occu_mask * ref_mask)) / occu_mask_avg

        return ssim_loss + pixel_loss

    def image_similarity_add_occu_mask(self, x, y, occu_mask):
        proj_error = ((1 - self.opt.alpha_recon_image) * tf.abs(x - y) * occu_mask)
        ssim_error = self.opt.alpha_recon_image * self.SSIM(x * occu_mask, y * occu_mask)
        return ssim_error + proj_error

    def image_similarity(self, x, y):
        return self.opt.alpha_recon_image * self.SSIM(x, y) + (1 - self.opt.alpha_recon_image) * tf.abs(x - y)

    def scale_pyramid(self, img, num_scales):
        if img == None:
            return None
        else:
            scaled_imgs = [img]
            _, h, w, _ = img.get_shape().as_list()
            for i in range(num_scales - 1):
                ratio = 2 ** (i + 1)
                nh = int(h / ratio)
                nw = int(w / ratio)
                scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
            return scaled_imgs

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def compute_smooth_loss(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

    def preprocess_image(self, image):
        # Assuming input image is uint8
        if image == None:
            return None
        else:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image * 2. - 1.

    def generate_feature_images(self, now_poses, optical_flow):
        # warping src images to tgt use current pose to generate feature images
        opt = self.opt
        bs = opt.batch_size
        fwd_rigid_flow_pyramid_mid = []
        for i in range(opt.num_source):
            fwd_rigid_flow = compute_rigid_flow_quater_change(tf.squeeze(tf.stop_gradient(self.pred_depth[0][:bs]), axis=3),
                                                              now_poses[i], self.intrinsics[:, 0, :, :],
                                                              False)  # tgt depth
            if not i:
                fwd_rigid_flow_concat = fwd_rigid_flow
            else:
                fwd_rigid_flow_concat = tf.concat([fwd_rigid_flow_concat, fwd_rigid_flow], axis=0)
        fwd_rigid_flow_pyramid_mid.append(fwd_rigid_flow_concat)

        fwd_rigid_warp_pyramid_mid = flow_warp(self.src_image_pyramid[0], fwd_rigid_flow_pyramid_mid[0])
        images_mid = tf.concat([self.tgt_image, fwd_rigid_warp_pyramid_mid, optical_flow], axis=3)

        return images_mid

    def generate_feature_images_2(self, now_poses):
        # warping src images to tgt use current pose to generate middle feature images
        opt = self.opt
        bs = opt.batch_size
        fwd_rigid_flow_pyramid_mid = []
        for i in range(opt.num_source):
            fwd_rigid_flow = compute_rigid_flow_quater_change(tf.squeeze(tf.stop_gradient(self.pred_depth[0][:bs]), axis=3),
                                                              now_poses[i], self.intrinsics[:, 0, :, :],
                                                              False)  # tgt depth
            if not i:
                fwd_rigid_flow_concat = fwd_rigid_flow
            else:
                fwd_rigid_flow_concat = tf.concat([fwd_rigid_flow_concat, fwd_rigid_flow], axis=0)
        fwd_rigid_flow_pyramid_mid.append(fwd_rigid_flow_concat)

        fwd_rigid_warp_pyramid_mid = flow_warp(self.src_image_pyramid[0], fwd_rigid_flow_pyramid_mid[0])
        # generate middle optical_flow
        with tf.variable_scope('train_flow', reuse=True):

            feature_mid = feature_pyramid_flow(fwd_rigid_warp_pyramid_mid, reuse=True)
            # fwd t-->s',  bwd s'-->t
            fwd_optical_flow_mid = construct_model_pwc_full(self.tgt_image, fwd_rigid_warp_pyramid_mid,
                                                            self.feature_tgt, feature_mid, reuse=True)
            bwd_optical_flow_mid = construct_model_pwc_full(fwd_rigid_warp_pyramid_mid, self.tgt_image, feature_mid,
                                                            self.feature_tgt, reuse=True)

        images_mid = tf.concat([self.tgt_image, fwd_rigid_warp_pyramid_mid, fwd_optical_flow_mid[0][:bs]], axis=3)
        return images_mid, fwd_optical_flow_mid, bwd_optical_flow_mid
