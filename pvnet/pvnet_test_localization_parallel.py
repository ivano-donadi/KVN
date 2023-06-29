from tools.pvnet_config import cfg, make_cfg, setup_cfg
import numpy as np
import os
import cv2
import sys
import torch
import queue
import argparse
from PIL import Image

from os import listdir
from os.path import isfile, join

from tools.inference_parallel import Inference

if __name__ == '__main__':	
    parser = argparse.ArgumentParser(description='Locate an object from an input image')
    
    parser.add_argument('-l', '--model',  
                        help='KVNet trained model', required=True)
    parser.add_argument('-f', '--meta_file', 
                        help='PVNet inference meta file (e.g., inference_meta.yaml)', required=True)
    parser.add_argument('-i', '--image', 
                        help='Input image or folder (e.g., image.jpg or images/)', required=True)
    parser.add_argument("--cfg_file", 
                        help='Low level configuration file, DO NOT CHANGE THIS PARAMETER IF YOU ARE NOT SURE (default = configs/custom_dsac.yaml)', 
                        default="configs/custom_dsac.yaml", type=str)
    
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = make_cfg(args)
    setup_cfg(cfg)
    
    inference = Inference(args.left_model, args.meta_file, cfg.heads.vote_dim, cfg.heads.seg_dim, True)
    
    if(os.path.isdir(args.image)):
      files = [f for f in listdir(args.image) if isfile(join(args.image, f))]
      for fimage in files:
        # read couple directly from left image
        if '_R' in fimage or '.exr' in fimage or 'txt' in fimage or 'mask' in fimage or 'border' in fimage:
          continue
        print(fimage)
        pos = inference(join(args.image, fimage))
        print(pos)
    else:
      pos = inference(args.image)
      print(pos)
        
