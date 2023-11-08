import numpy as np
import os
import cv2
import sys
import torch
import queue
from PIL import Image
from lib.datasets import make_data_loader
from lib.networks.pvnet.resnet18  import get_res_pvnet
from lib.datasets.transforms import make_transforms
from lib.utils import img_utils
from tools.handle_custom_dataset import read_inference_meta_stereo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils, pvnet_config
import lib.evaluators.custom.eval_utils as eval_utils


mean = pvnet_config.mean
std = pvnet_config.std

class Inference(object):

    def __init__(self, trained_model_fn, dataset_dir, cfg, vis=False ) :
      self.network = get_res_pvnet(cfg.heads.vote_dim, cfg.heads.seg_dim, cfg.train, cfg.test.ransac_rounds).cuda()
      pretrained_model = torch.load(trained_model_fn)
      self.network.load_state_dict(pretrained_model['net'], True)
      self.network.eval()

      #cfg.test.dataset_dir = dataset_dir
      self.data_loader = make_data_loader(cfg, is_train=False, json_fn="test.json", LR_split='')
      self.data_iter = iter(self.data_loader)
      self.dataset = self.data_loader.dataset
      self.vis= vis
      self.id = 0

    def __call__(self) :

      try:
        batch = next(self.data_iter)
      except:
        return None

      #img_id  = self.dataset.img_ids[self.id]
      #print("Image id: ",img_id)
      #anno = self.dataset.coco.loadAnns(self.dataset.coco.getAnnIds(imgIds=img_id))[0]
      #corner_3d = anno['corner_3d_L']

      kpt_3d = batch['kpt_3d'][0].numpy().astype(np.float64)
      K = batch['K'][0].numpy().astype(np.float64)
      baseline = batch['baseline'][0].item()

      for k in batch:
        if k != 'meta' and not isinstance(batch[k], list):
            batch[k] = batch[k].cuda()

      with torch.no_grad():
          output = self.network(batch=batch)
      
      offset_L = batch['offset_L'][0,None,:].detach().cpu().numpy()
      offset_R = batch['offset_R'][0,None,:].detach().cpu().numpy()
      kpt_2d_L = (output['kpt_2d_L'][0].detach().cpu().numpy() + offset_L).astype(np.float32)
      kpt_2d_R = (output['kpt_2d_R'][0].detach().cpu().numpy() + offset_R).astype(np.float32)
      var_L = output['var_L'][0].detach().cpu().numpy()
      var_R = output['var_R'][0].detach().cpu().numpy()
      mask_L = output['mask_L'][0].detach().cpu().numpy()

      initial_guess, _ = eval_utils.initial_guess_from_variance(var_L, kpt_2d_L, kpt_3d, K)

      try:
          var_L = np.linalg.inv(var_L)
      except:
          var_L = var_L
          var_R = np.full(var_R.shape,np.eye(var_R.shape[1]))
      try:
          var_R = np.linalg.inv(var_R)
      except:
          var_R = var_R
          var_L = np.full(var_L.shape,np.eye(var_L.shape[1]))

      R, t = pvnet_pose_utils.iterative_pnp(K, baseline, self.data_loader.dataset.height, self.data_loader.dataset.width, kpt_2d_L, kpt_2d_R, var_L, var_R, kpt_3d, initial_guess, True)

      pose_pred = np.c_[R,t]

      if self.vis :
        img = img_utils.unnormalize_img(batch['inp_L'][0].detach(), mean, std, False).cpu().numpy().transpose(1,2,0)
        kpt_3d_pred = batch['kpt_3d'][0].detach().cpu().numpy() 
        kpt_2d_pred = pvnet_pose_utils.project(kpt_3d_pred, K, pose_pred) - offset_L
        #corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred) - offset_L

        #pose_gt = batch['pose'][0].detach().cpu().numpy()
        #corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt) - offset_L

        self.visualize(img, mask_L, kpt_2d_pred, kpt_2d_L)

      self.id = self.id + 1
      return pose_pred
    
    def visualize( self, img, mask, kpt_2d_pred, kpt_2d_L):
    
      _, ax = plt.subplots(1)
      plt.title("Image id = ")
      ax.imshow(img)
      ax.imshow(mask, alpha=0.5)
      for kpt,kptp in zip(kpt_2d_pred, kpt_2d_L):
        ax.plot(kpt[0], kpt[1], marker = 'x', color = 'r')
        
        ax.plot(kptp[0], kptp[1], marker = 'x', color = 'g')
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
