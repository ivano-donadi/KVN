from yacs.config import CfgNode as CN
import os

cfg = CN()

# model
cfg.model = 'custom'
cfg.model_dir = 'data/model'

# network
cfg.network = 'res'

# network heads
cfg.heads = ''

# task
cfg.task = 'pvnet'

# gpus
cfg.gpus = [0, 1, 2, 3]

# if load the pretrained network
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 5
cfg.eval_ep = 5


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------

      
cfg.train = CN()

cfg.train.dataset_dir = 'data/custom'
cfg.train.epoch = 240
cfg.train.num_workers = 8

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-3-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.milestones = [20, 40, 60, 80, 100, 120, 160, 180, 200, 220]
cfg.train.gamma = 0.5

cfg.train.batch_size = 2

#augmentation
cfg.train.affine_rate = 0.
cfg.train.cropresize_rate = 0.
cfg.train.rotate_rate = 1.0
cfg.train.rotate_min_z = -30
cfg.train.rotate_max_z = 30

cfg.train.overlap_ratio = 0.8
cfg.train.resize_ratio_min = 0.8
cfg.train.resize_ratio_max = 1.2
cfg.train.negative_prob = 0.
cfg.train.image_sampler_minh = 256
cfg.train.image_sampler_minw = 256
cfg.train.image_sampler_maxh = 720
cfg.train.image_sampler_maxw = 1280
cfg.train.batch_sampler = 'image_size'

cfg.train.bg_prob = 0.5
cfg.train.kp_train_start = 149
cfg.train.max_x_tilt = 5
cfg.train.tod_crop_size = [126, 224, 30]
cfg.train.mirroring_prob = 0.5

cfg.train.kp_loss_threshold = 7.
cfg.train.kp_loss_beta = 1.
cfg.train.soft_ap_beta = 0.1
cfg.train.ransac_rounds = 256
cfg.train.target_entropy = 6
cfg.train.dsac_alpha = 0.1
cfg.train.dsac_beta = 100.
cfg.train.dsac_threshold = 0.001
cfg.train.use_epipolar = False
cfg.train.use_dsac = True

# test
cfg.test = CN()
cfg.test.dataset_dir = 'data/custom'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.icp = False
cfg.test.un_pnp = False
cfg.test.vsd = False
cfg.test.det_gt = False
cfg.test.ransac_rounds = 1024
cfg.test.batch_sampler = 'image_size'

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False

# dataset
cfg.cls_type = 'default'

# tless
cfg.tless = CN()
cfg.tless.pvnet_input_scale = (256, 256)
cfg.tless.scale_train_ratio = (1.8, 2.4)
cfg.tless.scale_ratio = 2.4
cfg.tless.box_train_ratio = (1.0, 1.2)
cfg.tless.box_ratio = 1.2
cfg.tless.rot = 360.
cfg.tless.ratio = 0.8

_heads_factory = {
    'pvnet': CN({'vote_dim': 18, 'seg_dim': 2}),
    'pvnet_stereo': CN({'vote_dim': 18, 'seg_dim': 2}),
    'pvnet_parallel': CN({'vote_dim': 18, 'seg_dim': 2}),
    'ct': CN({'ct_hm': 30, 'wh': 2})
}


def setup_cfg(cfg):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
#    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    if cfg.task in _heads_factory:
        cfg.heads = _heads_factory[cfg.task]

    # assign the network head conv
    cfg.head_conv = 64 if 'res' in cfg.network else 256

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task,'')
    cfg.record_dir = os.path.join(cfg.model_dir, 'record')
    cfg.result_dir = os.path.join(cfg.model_dir, 'result')


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    opts_idx = [i for i in range(0, len(args.opts), 2) if args.opts[i].split('.')[0] in cfg.keys()]
    opts = sum([[args.opts[i], args.opts[i + 1]] for i in opts_idx], [])
    cfg.merge_from_list(opts)
    return cfg
  
