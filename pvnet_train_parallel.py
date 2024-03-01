from tools.pvnet_config import cfg, make_cfg, setup_cfg
import torch
import os
import argparse
import json

from lib.networks.pvnet.resnet18  import get_res_pvnet
from lib.train.trainers.pvnet import NetworkWrapper
from lib.train.trainers.trainer import Trainer
from lib.train.recorder import Recorder
from lib.utils.optimizer.lr_scheduler import MultiStepLR
from lib.evaluators import make_evaluator
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network

def make_network(cfg):

    network = get_res_pvnet(cfg.heads.vote_dim, cfg.heads.seg_dim, cfg.train, cfg.test.ransac_rounds)
    trainer = NetworkWrapper(network, cfg.train)
    trainer = Trainer(trainer)

    params = []
    for key, value in network.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay}]
        


    return network, trainer, params

def make_optimizer(cfg, params):

    optimizer = torch.optim.Adam(params, cfg.train.lr, weight_decay=cfg.train.weight_decay)
    reset_threshold=0
    scheduler = MultiStepLR(optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma, reset_threshold=reset_threshold)
    recorder = Recorder(cfg)

    return optimizer, scheduler, recorder

def make_evaluators(cfg):
    test_evaluator = make_evaluator(cfg, dataset_dir=cfg.test.dataset_dir,is_train=False, suffix='', json_fn = 'test_val.json')
    train_evaluator = make_evaluator(cfg, dataset_dir=cfg.train.dataset_dir, is_train=True, suffix='')
        

    return test_evaluator, train_evaluator

def make_data_loaders(cfg):

    print('Valid')
    val_loader = make_data_loader(cfg, is_train=False, max_iter=cfg.ep_iter, bkg_imgs_dir=cfg.bkg_imgs_dir, json_fn='test_val.json', LR_split='WwWwW')
    print('Train')
    train_loader = make_data_loader(cfg, is_train=True, max_iter=cfg.ep_iter, bkg_imgs_dir=cfg.bkg_imgs_dir, LR_split='')

    return val_loader, train_loader


def main(cfg):

    if not os.path.isdir(cfg.model_dir) :
      os.mkdir(cfg.model_dir)
    
    best_model_dir = os.path.join(cfg.model_dir, 'best_model')
    
    if not os.path.isdir(best_model_dir) :
      os.mkdir(best_model_dir)
      bestepoch = 0
    else:
      if len(os.listdir(best_model_dir)) < 1:
          bestepoch = 0
      else:
          bestepoch = [int(x.split(".")[0]) for x in os.listdir(best_model_dir) if x.endswith(".pth")][0]
      
    eval_stats_fn = os.path.join(cfg.model_dir, 'eval_stats.json')
    
    if not cfg.resume :
      if os.path.exists(eval_stats_fn):
        os.remove(eval_stats_fn)
    
    network, trainer, params = make_network(cfg)
    optimizer, scheduler, recorder = make_optimizer(cfg, params)
    test_evaluator, _ = make_evaluators(cfg)
    validation_loader, train_loader = make_data_loaders(cfg)

    begin_epochs = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume, suffix='')

    if not recorder.best_loss:
        recorder.best_loss = None
        recorder.best_loss_epoch = 0

    for epoch in range(begin_epochs, cfg.train.epoch):

        #early stop
        if epoch - bestepoch > 50:
            print("No validation improvement for 50 epochs: early stopping")
            break

        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

#        train_loss = recorder.loss_stats['loss']
#        if recorder.best_loss is None:
#            recorder.best_loss = train_loss.avg
#            recorder.best_loss_epoch = epoch
#        else:
#            diff = train_loss.avg - recorder.best_loss
#            if diff < 1e-4:
#                recorder.best_loss = train_loss.avg
#                recorder.best_loss_epoch = epoch
#            if diff >= 1e-4 and (epoch - recorder.best_loss_epoch) > 10:
#                print("No training loss improvement for 10 epochs: early_stopping")
#                break


        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir, suffix='')

        if (epoch + 1) % cfg.eval_ep == 0:

            print("Evaluation on validation set:")
            new_stats, loss_val = trainer.val(epoch, validation_loader, test_evaluator, recorder)
                
            if os.path.exists(eval_stats_fn):
                with open(eval_stats_fn, 'r') as eval_stats:
                    old_stats = json.load(eval_stats)
            else:
                old_stats = {'proj2d': 0.0, 'add': 0.0} 
            
            new_score = (new_stats['proj2d'] + new_stats['add'])/2;
            old_score = (old_stats['proj2d'] + old_stats['add'])/2;

                

            if new_score > old_score :
                # split at suffix underscore
                filetype = '.pth'
                pths = [int(pth.split('.')[0]) for pth in os.listdir(best_model_dir)  if filetype in pth]
                if pths :
                    os.system('rm {}'.format(os.path.join(best_model_dir, ('{}' + filetype).format(min(pths)))))
                save_model(network, optimizer, scheduler, recorder, epoch, best_model_dir, suffix = '')

                # do not overwrite stats from other net
                new_proj = new_stats['proj2d']
                new_add = new_stats['add']
                new_stats = old_stats
                new_stats['proj2d'] = new_proj
                new_stats['add'] = new_add
                with open(eval_stats_fn, 'w') as eval_stats:
                    json.dump(new_stats, eval_stats)

                bestepoch = epoch


    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='KVNet training tool',
                                     epilog="You need at least to provide an input training dataset "
                                            "and to specify the output directory where the trained models will be stored."
                                            "The best model checkpoint will be stored inside the best_model subdirectory")

    parser.add_argument('-d', '--dataset_dir', 
                        help='Input directory containing the training dataset', 
                        required=True)    
    parser.add_argument('-m', '--model_dir',  
                        help='Output directory where the trained models will be stored', required=True)
    parser.add_argument('-b', '--batch_size', 
                        help='Number of training examples in one forward/backward pass (default = 2)', type=int, default = 2)
    parser.add_argument('-n', '--num_epoch', 
                        help='Number of epochs to train  (default = 240)', type=int, default = 240)
    parser.add_argument('-e', '--eval_ep', 
                        help='Number of epochs after which to evaluate (and eventually save) the model (default = 5)', type=int, default = 5)
    parser.add_argument('-s', '--save_ep', 
                        help='Number of epochs after which to save the model (default = 5)', type=int, default = 5)
    parser.add_argument('--bkg_imgs_dir',
                        help='Optional background images directory, to be used to augment the dataset', default = '')
    parser.add_argument('--disable_resume', action='store_true', help='If specified, disable train resume and start a new train')
    parser.add_argument("--cfg_file", 
                        help='Low level configuration file, DO NOT CHANGE THIS PARAMETER IF YOU ARE NOT SURE (default = configs/custom_dsac.yaml)', 
                        default="configs/custom_dsac.yaml", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()

    cfg = make_cfg(args)

    cfg.model_dir = args.model_dir;
    # Same dataset dir for both train and test
    cfg.test.dataset_dir = cfg.train.dataset_dir = args.dataset_dir;
    cfg.train.batch_size = args.batch_size;

    cfg.train.epoch = args.num_epoch;
    cfg.eval_ep = args.eval_ep;
    cfg.save_ep = args.save_ep;
    cfg.resume = not args.disable_resume
    cfg.bkg_imgs_dir = args.bkg_imgs_dir
    
    setup_cfg(cfg)
    
    main(cfg)
