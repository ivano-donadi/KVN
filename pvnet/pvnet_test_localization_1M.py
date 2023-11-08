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
    
    parser.add_argument('-d', '--dataset_dir', 
                        help='Input directory containing the test dataset', required=True)    
    parser.add_argument('-m', '--model',  
                        help='KVNet trained model', required=True)
    parser.add_argument('--cls_type', default='micro_plate', help='Object class of interest [default: ]')
    parser.add_argument("--cfg_file", 
                        help='Low level configuration file, DO NOT CHANGE THIS PARAMETER IF YOU ARE NOT SURE (default = configs/custom_dsac.yaml)', 
                        default="configs/custom_dsac.yaml", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = make_cfg(args)

    cfg.data = args.dataset_dir
    cfg.image_height = 480
    cfg.image_width = 480
    cfg.split = 'train'
    cfg.cls_type = args.cls_type
    cfg.num_kp = 9

    setup_cfg(cfg)

    setup_cfg(cfg)
    
    inference = Inference(args.model, args.dataset_dir, cfg, True)

    pose = []

    while pose is not None:
      pose = inference()
      print('pose=\n',pose)
        
