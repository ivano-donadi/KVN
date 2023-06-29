from tools.pvnet_config import cfg, make_cfg, setup_cfg
import torch
import os
import argparse
import json
import tqdm

from lib.networks.pvnet.resnet18  import get_res_pvnet
from lib.evaluators.custom.pvnet_parallel import Evaluator
from lib.datasets import make_data_loader


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='KVNet evaluation tool',
                                     epilog="You need at least to provide an (annotated) input test dataset "
                                            "and a KVNet trained model")

    parser.add_argument('-d', '--dataset_dir', 
                        help='Input directory containing the test dataset', required=True)    
    parser.add_argument('-m', '--model',  
                        help='KVNet trained model', required=True)
    parser.add_argument("--cfg_file", 
                        help='Low level configuration file, DO NOT CHANGE THIS PARAMETER IF YOU ARE NOT SURE (default = configs/custom_dsac.yaml)', 
                        default="configs/custom_dsac.yaml", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    
    cfg = make_cfg(args)

    cfg.test.dataset_dir = args.dataset_dir;

    setup_cfg(cfg)
      
    network = get_res_pvnet(cfg.heads.vote_dim, cfg.heads.seg_dim, cfg.train, cfg.test.ransac_rounds).cuda()
    pretrained_model = torch.load(args.model)
    network.load_state_dict(pretrained_model['net'], True)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False, json_fn="test.json", LR_split='')
    evaluator = Evaluator(cfg.result_dir, cfg.test.dataset_dir, is_train=False)


    for batch in tqdm.tqdm(data_loader):
        
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()

        with torch.no_grad():
            output = network(batch=batch)
        evaluator.joint_evaluate(output,batch)
    evaluator.summarize()
