import sys, getopt
from lib.config import cfg, args
import numpy as np

def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize(output, batch)


def print_help(cmd_name):
    print ('python3 ',cmd_name, ' --cfg_file <cfg_file> -i <inputimage>')
    print ('python3 ',cmd_name, ' --cfg_file <cfg_file> -d <inputdir>')

def main(argv):
    inputimage = ''
    inputdir = ''
    if len(argv) < 2:
        print_help(argv[0])
        sys.exit()
    try:
        opts, args = getopt.getopt(argv,"hi:d:")
    except:
        print_help()
        raise
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt in ("-i"):
            inputimage = arg
            print ('Input image is', inputimage)
        elif opt in ("-d"):
            inputdir = arg
            print ('Input images directory is', inputdir)

if __name__ == "__main__":
    print(cfg)
    main(sys.argv)
