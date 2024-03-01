from tools.pvnet_config import cfg, make_cfg, setup_cfg
import torch
import os
import argparse
import json
import tqdm
import numpy as np

from lib.networks.pvnet.resnet18  import get_res_pvnet
from lib.evaluators import make_evaluator
from lib.datasets import make_data_loader


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='KVN evaluation tool',
                                     epilog="You need at least to provide an (annotated) input test dataset "
                                            "and a KVN trained model")

    parser.add_argument('-d', '--dataset_dir', 
                        help='Input directory containing the test dataset', required=True)    
    parser.add_argument('-m', '--model',  
                        help='KVN trained model', required=True)
    parser.add_argument('-o', '--output_dir',  
                        help='Optional output dir where to save the results')
    parser.add_argument('--num_iters',  
                        help='Number of evaluation iterations to average over (default=10)', default=10)
    parser.add_argument("--cfg_file", 
                        help='Low level configuration file, DO NOT CHANGE THIS PARAMETER IF YOU ARE NOT SURE (default = configs/custom_dsac.yaml)', 
                        default="configs/custom_dsac.yaml", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    
    cfg = make_cfg(args)

    cfg.test.dataset_dir = args.dataset_dir;
    num_iters = args.num_iters
    setup_cfg(cfg)
      
    network = get_res_pvnet(cfg.heads.vote_dim, cfg.heads.seg_dim, cfg.train, cfg.test.ransac_rounds).cuda()
    pretrained_model = torch.load(args.model)
    network.load_state_dict(pretrained_model['net'], True)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False, json_fn="test.json", LR_split='')


    proj2d_sum = 0
    add_sum = 0
    cm2_sum = 0
    iters = 0
    while iters < num_iters:
      evaluator = make_evaluator(cfg, dataset_dir=cfg.test.dataset_dir, is_train=False, suffix='', json_fn = 'test.json')
      for batch in tqdm.tqdm(data_loader):
        
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()

        with torch.no_grad():
            output = network(batch=batch)
        evaluator.joint_evaluate(output,batch)
    
      cur_proj2d = np.mean(evaluator.proj2d_it)
      cur_add = np.mean(evaluator.add_it)
      cur_cm2 = np.mean(evaluator.cm2_it)

      proj2d_sum += cur_proj2d
      add_sum += cur_add
      cm2_sum += cur_cm2
      iters += 1
      #evaluator.summarize()
      
    if args.output_dir is not None :
      result_file_name = os.path.join(args.output_dir, 'test.txt')
      try:
          with open(result_file_name, 'w') as file:
            file.write('Final 2d projections metric (iterative pnp): {}\n'.format(proj2d_sum/num_iters))
            file.write('Final ADD metric (iterative PnP): {}\n'.format(add_sum/num_iters))
            file.write('Final <2cm metric (iterative pnp): {}\n'.format(cm2_sum/num_iters))  
            file.write('{}\t{}\t{}\n'.format(proj2d_sum/num_iters,add_sum/num_iters,cm2_sum/num_iters))  
            
      except Exception as e:
          print("Error:", e)
        
    print('Final 2d projections metric (iterative pnp): {}'.format(proj2d_sum/num_iters))
    print('Final ADD metric (iterative PnP): {}'.format(add_sum/num_iters))
    print('Final <2cm metric (iterative pnp):: {}'.format(cm2_sum/num_iters))
