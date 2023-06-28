from . import transforms_stereo as tfs
from . import transforms as tfp
from . import samplers
import torch
import torch.utils.data
import imp
import os
from .collate_batch import make_collator
import time
import numpy as np


torch.multiprocessing.set_sharing_strategy('file_system')



def _dataset_factory(data_source, task, ):
    module = '.'.join(['lib.datasets', data_source, task])
    path = os.path.join('lib/datasets', data_source, task+'.py')
    dataset = imp.load_source(module, path).Dataset
    return dataset


def make_dataset(cfg, dataset_dir, transforms, json_fn, is_train=True, LR_split=None):
    
    if(is_train) :
      args = { 'id' : 'custom',
               'data_root' : dataset_dir,
               'ann_file' : os.path.join(dataset_dir, json_fn),
               'split' :'train',
               'transforms' : transforms,
               'cfg': cfg }
    else :
      args = { 'id' : 'custom',
               'data_root' : dataset_dir,
               'ann_file' : os.path.join(dataset_dir, json_fn),
               'split' : 'test',
               'transforms' : transforms,
               'cfg': cfg }

    if LR_split:
        args['suffix'] = LR_split

    data_source = args['id']
    dataset = _dataset_factory(data_source, cfg.task)
    del args['id']
    dataset = dataset(**args)
    return dataset


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter, is_train):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)

    strategy = cfg.train.batch_sampler if is_train else cfg.test.batch_sampler
    if strategy == 'image_size':
        batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size, drop_last, cfg.train.image_sampler_minh, cfg.train.image_sampler_minw, cfg.train.image_sampler_maxh, cfg.train.image_sampler_maxw)

    return batch_sampler


def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2 ** 16))))


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1, bkg_imgs_dir = "", json_fn='train.json', LR_split=None):
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = True
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset_dir = cfg.train.dataset_dir if is_train else cfg.test.dataset_dir

    if cfg.task == 'pvnet_stereo':
        transforms = tfs.make_transforms(is_train, bkg_imgs_dir, cfg.train.bg_prob)
    else:
        transforms = tfp.make_transforms(is_train, bkg_imgs_dir, cfg.train.bg_prob)

    dataset = make_dataset(cfg, dataset_dir, transforms, json_fn, is_train, LR_split)
    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter, is_train)
    num_workers = cfg.train.num_workers
    collator = make_collator(cfg)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        worker_init_fn=worker_init_fn
    )

    return data_loader
    
data_loader = torch.utils.data.DataLoader
