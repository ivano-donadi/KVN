#from lib.datasets.dataset_catalog import DatasetCatalog
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_pose_utils, pvnet_data_utils
import os
from lib.utils.linemod import linemod_config
import torch
from PIL import Image
from lib.utils.img_utils import read_depth
from scipy import spatial
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

from lib.utils.pvnet.visualize_utils import visualize_dsac_results, visualize_dsac_results_split, visualize_tod_ann, visualize_error_maps, visualize_3d_bbox

from . import eval_utils

class Evaluator:

    def __init__(self, result_dir, dataset_dir, cls_type = 'default', is_train=True, suffix = '_L', json_fn = None):
        self.result_dir = result_dir
        data_root = dataset_dir
        self.cls_type = cls_type
        
        if is_train:
            self.ann_file = os.path.join(data_root, 'train.json')
        else:
            self.ann_file = os.path.join(data_root, 'test.json')
        
        if json_fn is not None:
            self.ann_file = os.path.join(data_root, json_fn)
            
        self.coco = coco.COCO(self.ann_file)

        model_path = os.path.join(data_root, 'model.ply')
        self.model = pvnet_data_utils.get_ply_model(model_path)
        self.diameter = np.loadtxt(os.path.join(data_root, 'diameter.txt')).item()

        self.proj2d = []
        self.proj2d_it = []
        self.add = []
        self.mae = []
        self.cm2 = []
        self.icp_add = []
        self.cmd5 = []
        self.cmd5_it = []
        self.mask_ap = []
        self.add_it = []
        self.mae_it = []
        self.cm2_it = []
        self.failed_count = 0
        self.icp_render = None
        self.suffix = suffix

    def evaluate(self, output, batch):
        offset = batch['offset'+self.suffix][0,None,:].detach().cpu().numpy()
        kpt_2d = output['kpt_2d'+self.suffix][0].detach().cpu().numpy() + offset
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        obj_name = anno['cls']
        kpt_3d = np.concatenate([anno['fps_3d'+self.suffix], [anno['center_3d'+self.suffix]]], axis=0)
        K = np.array(anno['K'])
        #print(kpt_2d.shape, kpt_3d.shape)
        pose_gt = np.array(anno['pose'+self.suffix])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        self.proj2d.append(eval_utils.projection_2d(self.model, pose_pred, pose_gt, K))
        if obj_name in ['ball_0', 'bottle_0', 'bottle_1', 'bottle_2', 'cup_0', 'cup_1']:
            mean_dist, add = eval_utils.add_metric(self.model, self.diameter, pose_pred, pose_gt, syn= True)
        else:
            mean_dist, add = eval_utils.add_metric(self.model, self.diameter, pose_pred, pose_gt, syn = False)
        self.add.append(add)
        self.cm2.append(mean_dist < 0.02)
        self.mae.append(mean_dist)
        self.cmd5.append(eval_utils.cm_degree_5_metric(pose_pred, pose_gt))
        self.mask_ap.append(eval_utils.mask_iou(output, batch))

    def joint_evaluate(self, output, batch):
        self.suffix = '_L'
        self.evaluate(output, batch)
        #self.suffix = '_R'
        #self.evaluate(output_R, batch_R)

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        K = np.array(anno['K'])
        baseline = anno['baseline']

        pose_gt_L = np.array(anno['pose_L'])
        obj_name = anno['cls']

        syn = obj_name in ['ball_0', 'bottle_0', 'bottle_1', 'bottle_2', 'cup_0', 'cup_1']

        offset_L = batch['offset_L'][0,None,:].detach().cpu().numpy()
        offset_R = batch['offset_R'][0,None,:].detach().cpu().numpy()
        kpt_2d_L = output['kpt_2d_L'][0].detach().cpu().numpy() + offset_L
        kpt_2d_R = output['kpt_2d_R'][0].detach().cpu().numpy() + offset_R
        var_L = output['var_L'][0].detach().cpu().numpy()
        var_R = output['var_R'][0].detach().cpu().numpy()
        kpt_3d_L = np.concatenate([anno['fps_3d_L'], [anno['center_3d_L']]], axis=0)
        kpt_3d_R = np.concatenate([anno['fps_3d_R'], [anno['center_3d_R']]], axis=0)

        # skip ball_0 image with object completely out of camera view (check for centroid out of image bounds)
        #print(batch['kpt_2d_L'][0][-1][1], offset_L[0][1])
        if obj_name == 'ball_0' and batch['kpt_2d_L'][0][-1][1] > 126:
            print("skipping emplty ball_0 image")
            return

        print(kpt_3d_L.shape, kpt_3d_L.dtype, K.dtype)

        initial_guess_L, guess_cost_L = eval_utils.initial_guess_from_variance(var_L, kpt_2d_L, kpt_3d_L, K)
        initial_guess_R, guess_cost_R = eval_utils.initial_guess_from_variance(var_R, kpt_2d_R, kpt_3d_R, K)
        if guess_cost_L < guess_cost_R:
            initial_guess = initial_guess_L
        else:
            initial_guess = initial_guess_R
            initial_guess[0,3] -= baseline

        mae, cm2, cmd5, add, proj2d, pose_pred_it = eval_utils.it_pnp_metrics(self.model, self.diameter, K, baseline, 720, 1280, kpt_2d_L, kpt_2d_R, kpt_3d_L, var_L, var_R, pose_gt_L,syn=syn, initial_guess = initial_guess, obj_name = obj_name)

        good_mask = eval_utils.mask_iou(output, batch)
        if False: #not good_mask and mae > 0.05: 
            visualize_tod_ann(batch['inp_L'][0].detach().cpu() ,batch['inp_L'][0].detach().cpu(),  batch['kpt_2d_L'][0].detach().cpu(),  output['kpt_2d_L'][0].detach().cpu(), batch['mask_L'][0].detach().cpu(),output['mask_L'][0].detach().cpu())
            #visualize_tod_ann(batch['inp_R'][0].detach().cpu() ,batch['inp_R'][0].detach().cpu(),  batch['kpt_2d_R'][0].detach().cpu(),  output['kpt_2d_R'][0].detach().cpu(), batch['mask_R'][0].detach().cpu(),output['mask_R'][0].detach().cpu())
            #visualize_3d_bbox(batch['inp_L'][0], anno['corner_3d_L'], K, pose_pred_it, pose_gt_L, offset_L)
            #visualize_error_maps(batch['vertex_L'], output['vertex_L'], batch['kpt_2d_L'],batch['mask_L'])
            #probs = torch.softmax(output['scores_L'][0], dim=0)
            #visualize_dsac_results(batch['inp_L'][0], output['keypoints_L'][0], probs, batch['kpt_2d_L'][0].detach().cpu()) 
            #probs = torch.softmax(output['scores_R'][0], dim=0)
            #visualize_dsac_results(batch['inp_R'][0], output['keypoints_R'][0], probs, batch['kpt_2d_R'][0].detach().cpu()) 

        if False and mae > 1:
            self.failed_count += 1
        else:
            self.mae_it.append(mae)
            self.cm2_it.append(cm2)
            self.cmd5_it.append(cmd5)
            self.add_it.append(add)
            self.proj2d_it.append(proj2d)

    def summarize(self):
        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        cmd5 = np.mean(self.cmd5)
        cm2 = np.mean(self.cm2)
        ap = np.mean(self.mask_ap)
        mae = np.mean(self.mae)
        curve, step = eval_utils.curve_from_mae(self.mae)
        auc = eval_utils.auc_from_curve(curve,step)
        print('>> Monocular image evaluation: ')
        print('   2d projections metric: {}'.format(proj2d))
        print('   Failed images count:', self.failed_count)
        print('   ADD metric: {}'.format(add))
        print('   <2cm metric: {}'.format(cm2))
        print('   MAE metric: {}'.format(mae))
        print('   5 cm 5 degree metric: {}'.format(cmd5))
        print('   AUC metric: {}'.format(auc))
        print('   mask ap70: {}'.format(ap))
        joint = len(self.add_it) > 0
        if joint:
            mae_it = np.mean(self.mae_it)
            cm2_it = np.mean(self.cm2_it)
            cmd5_it = np.mean(self.cmd5_it)
            proj2d_it = np.mean(self.proj2d_it)
            it_add = np.mean(self.add_it)
            curve, step = eval_utils.curve_from_mae(self.mae_it)
            auc = eval_utils.auc_from_curve(curve, step)
            print('>> Stereo image evaluation: ')
            print('   2d projections metric (iterative pnp): {}'.format(proj2d_it))
            print('   ADD metric (iterative PnP): {}'.format(it_add))
            print('   <2cm metric (iterative pnp): {}'.format(cm2_it))
            print('   MAE metric (iterative pnp): {}'.format(mae_it))
            print('   5 cm 5 degree metric (iterative pnp): {}'.format(cmd5_it))
            print('   AUC metric (iterative pnp): {}'.format(auc))
            
            #default_x_ticks = range(len(curve))
            #plt.plot(default_x_ticks, curve)
            #plt.show()


        self.proj2d = []
        self.proj2d_it = []
        self.add = []
        self.mae = []
        self.cm2 = []
        self.icp_add = []
        self.cmd5 = []
        self.cmd5_it = []
        self.mask_ap = []
        self.add_it = []
        self.mae_it = []
        self.cm2_it = []
        self.failed_count = 0
        if joint: 
            return {'proj2d': proj2d_it, 'add': it_add, 'cmd5': cmd5_it, 'ap': ap, 'cm2': cm2_it, 'mae':mae_it}
        else:
            return {'proj2d': proj2d, 'add': add, 'cmd5': cmd5, 'ap': ap, 'cm2': cm2, 'mae': mae}
