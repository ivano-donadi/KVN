import numpy as np
import os
import cv2
import sys
import torch
import queue
from PIL import Image

from lib.networks.pvnet.resnet18  import get_res_pvnet
from lib.datasets.transforms import make_transforms

from tools.handle_custom_dataset import read_inference_meta_stereo


import matplotlib.pyplot as plt
import matplotlib.patches as patches

from lib.utils.pvnet import pvnet_pose_utils



class Inference(object):

    def __init__(self, trained_model_fn, inference_meta_fn, 
                 vote_dim = 18, seg_dim = 2, vis=False ) :

      self.network = get_res_pvnet(vote_dim, seg_dim).cuda()
      pretrained_model = torch.load(trained_model_fn)
      self.network.load_state_dict(pretrained_model['net'], True)
      self.network.eval()
      
      K, baseline, corner_3d, center_3d, fps_3d = read_inference_meta_stereo( inference_meta_fn )
      
      self.K = np.array(K)
      self.baseline = baseline
      self.kpt_3d = np.concatenate([fps_3d, np.transpose(center_3d)], axis=0)
      self.corner_3d = np.array(corner_3d)

      # Workaround to use original PvNet functions that provide also kpts
      # and mask as output 
      self.transforms = make_transforms(is_train=False)
      
      self.vis= vis

    def __call__(self, img) :

      if isinstance(img, str) :
        img_L = Image.open(img)
        img_R = Image.open(img.replace('L','R'))
      
      transf_img_L, kpts_L, mask_L = self.transforms(img_L)
      transf_img_R, kpts_R, mask_R = self.transforms(img_R)
      
      data_L = torch.tensor([transf_img_L])
      data_L = data_L.cuda()
      data_R = torch.tensor([transf_img_R])
      data_R = data_R.cuda()

      batch = {'inp_L': data_L, 'inp_R': data_R}

      with torch.no_grad():
        output = self.network(batch)
      
      kpt_2d_L = output['kpt_2d_L'][0].detach().cpu().numpy()
      kpt_2d_R = output['kpt_2d_R'][0].detach().cpu().numpy()

      R, t = pvnet_pose_utils.iterative_pnp(self.K, self.baseline, img_L.size[0], img_L.size[1], kpt_2d_L, kpt_2d_R, self.kpt_3d)

      pose_pred = np.c_[R,t]

      if self.vis :
        self.visualize( img_L, pose_pred )

      return pose_pred
    
    def visualize( self, img, pose_pred ) :
      
      corner_2d_pred = pvnet_pose_utils.project(self.corner_3d, self.K, pose_pred)
      _, ax = plt.subplots(1)
      ax.imshow(img)
      ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
      ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
      plt.show()  
      
      
class MultiInference(object):

    def __init__(self, vote_dim = 18, seg_dim = 2 ) :
                   
      self.inference = list()
      self.vote_dim = vote_dim
      self.seg_dim = seg_dim
      
    def addModel(self, trained_model_fn, inference_meta_fn ) :
      self.inference.append(Inference(trained_model_fn, inference_meta_fn, self.vote_dim, self.seg_dim, False))

    def __call__(self, img, index) :
      
      return self.inference[index](img)
