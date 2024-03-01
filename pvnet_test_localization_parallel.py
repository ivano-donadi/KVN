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
                        help='KVN trained model', required=True)
    parser.add_argument("--cfg_file", 
                        help='Low level configuration file, DO NOT CHANGE THIS PARAMETER IF YOU ARE NOT SURE (default = configs/custom_dsac.yaml)', 
                        default="configs/custom_dsac.yaml", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = make_cfg(args)
    setup_cfg(cfg)
    
    inference = Inference(args.model, args.dataset_dir, cfg, True)

    pose = []

    while pose is not None:
      pose = inference()
      print('pose=\n',pose)
        
