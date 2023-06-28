
import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image
from lib.utils.pvnet import pvnet_data_utils, pvnet_linemod_utils, visualize_utils
from lib.utils.linemod import linemod_config
from lib.datasets.augmentation import _augmentation_keep_epipolar_constraints
import random
import torch
import cv2
import time


class Dataset(data.Dataset):

    def __init__(self, ann_file, data_root, split, cfg, transforms=None, suffix = '_L'):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms
        self.suffix = suffix
        self.cfg = cfg

    def read_data_suffix(self, anno, id, suffix):
        path = self.coco.loadImgs(int(id))[0]['file_name'+suffix]
        inp = Image.open(path)
        kpt_2d = np.concatenate([anno['fps_2d'+suffix], [anno['center_2d'+suffix]]], axis=0)

        #cls_idx = linemod_config.linemod_cls_names.index(anno['cls']) + 1
        if anno['mask_path'+suffix] == 'dummy_mask' :
          mask = []
        else:
          mask = pvnet_data_utils.read_linemod_mask(anno['mask_path'+suffix], anno['type'],1)

        K = np.array(anno['K']).astype(np.float32)
        baseline = anno['baseline']
        # same for both
        if suffix == '_L':
            kpt_3d = np.concatenate([anno['fps_3d_L'], [anno['center_3d_L']]], axis=0).astype(np.float32)
            pose = np.array(anno['pose'+suffix]).astype(np.float32)
        else:
            kpt_3d = None
            pose = None
        res = {'inp': inp, 'kpt_2d': kpt_2d, 'mask': mask, 'path':path, 'K': K, 'baseline': baseline, 'kpt_3d': kpt_3d, 'pose': pose}

        return res

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        res_L = self.read_data_suffix(anno, img_id, '_L')
        res_R = self.read_data_suffix(anno, img_id, '_R')

        return res_L, res_R

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.img_ids[index]
        
        res_L, res_R = self.read_data(img_id)
        K = res_L['K']
        baseline = res_L['baseline']
        pose = res_L['pose']
        inp_L = np.asarray(res_L['inp']).astype(np.uint8)
        inp_R = np.asarray(res_R['inp']).astype(np.uint8)
        mask_L = res_L['mask']
        mask_R = res_R['mask']
        kpt_2d_L = res_L['kpt_2d']
        kpt_2d_R = res_R['kpt_2d']
        kpt_3d = res_L['kpt_3d'] 

        if self.split == 'train':
            #t = time.process_time()
            inp_L, inp_R, mask_L, mask_R, kpt_2d_L, kpt_2d_R = self.augment(inp_L, inp_R, mask_L, mask_R, kpt_2d_L, kpt_2d_R, K,'')
            #print('time: ',time.process_time()-t,'s')
 

        inp_L = inp_L.transpose(2,0,1)
        inp_R = inp_R.transpose(2,0,1)

        inp_L, offset_L = self.do_occlude_crop(inp_L, kpt_2d_L, self.cfg.train.tod_crop_size)
        mask_L, _ = self.do_occlude_crop(mask_L, kpt_2d_L, self.cfg.train.tod_crop_size)
        kpt_2d_L -= offset_L[None,:]

        inp_R, offset_R = self.do_occlude_crop(inp_R, kpt_2d_R, self.cfg.train.tod_crop_size)
        mask_R, _ = self.do_occlude_crop(mask_R, kpt_2d_R, self.cfg.train.tod_crop_size)
        kpt_2d_R -= offset_R[None,:]

        inp_L = inp_L.transpose(1,2,0)
        inp_R = inp_R.transpose(1,2,0)

        if self._transforms is not None:
            inp_L, kpt_2d_L, mask_L = self._transforms(inp_L, kpt_2d_L, mask_L)
            inp_R, kpt_2d_R, mask_R = self._transforms(inp_R, kpt_2d_R, mask_R)    

        #visualize_utils.visualize_tod_ann(torch.from_numpy(inp_L),torch.from_numpy(inp_R),kpt_2d_L, kpt_2d_R, mask_L, mask_R,False)   

        vertex_L = pvnet_data_utils.compute_vertex(mask_L, kpt_2d_L).transpose(2, 0, 1)
        vertex_R = pvnet_data_utils.compute_vertex(mask_R, kpt_2d_R).transpose(2, 0, 1)

        kpt_2d_L = kpt_2d_L.astype(np.float32)
        kpt_2d_R = kpt_2d_R.astype(np.float32)
        mask_L = mask_L.astype(np.uint8)
        mask_R = mask_R.astype(np.uint8)

        ret = {'inp_L': inp_L, 'mask_L': mask_L, 'vertex_L': vertex_L, 'kpt_2d_L': kpt_2d_L, 'offset_L': offset_L, 
            'inp_R': inp_R, 'mask_R': mask_R, 'vertex_R': vertex_R, 'kpt_2d_R': kpt_2d_R, 'offset_R': offset_R,
            'K':K, 'baseline': baseline, 'pose':pose, 'kpt_3d': kpt_3d,
            'img_id': img_id, 'meta': {}}
        
        return ret

    def __len__(self):
        return len(self.img_ids)

    def augment(self, img_L, img_R, mask_L, mask_R, kpt_2d_L, kpt_2d_R, K, img_path):
        # add one column to kpt_2d for convenience to calculate
        img_L = np.asarray(img_L).astype(np.uint8)
        img_R = np.asarray(img_R).astype(np.uint8)
        hcoords_L = np.concatenate((kpt_2d_L, np.ones((kpt_2d_L.shape[0], 1))), axis=-1)
        hcoords_R = np.concatenate((kpt_2d_R, np.ones((kpt_2d_R.shape[0], 1))), axis=-1)
        foreground = np.sum(mask_L)
        if foreground > 0:
            img_L, img_R, mask_L, mask_R, hcoords_L, hcoords_R = _augmentation_keep_epipolar_constraints(img_L,img_R, mask_L, mask_R, hcoords_L, hcoords_R, K, self.cfg)
        else:
            print('|||||| No foreground for img',img_path,'|||||||||||||||||')

        kpt_2d_L = hcoords_L[:, :2]
        kpt_2d_R = hcoords_R[:, :2]

        return img_L, img_R, mask_L, mask_R, kpt_2d_L, kpt_2d_R


    def do_occlude_crop(self, image,
                    key_pts,
                    crop):
        """Crop area around the object."""
        # Crop is [W, H, R]', where 'R' is right-disparity offset; or else [].
        # Images can be either floating-point or uint8.

        offset = np.array([0, 0], dtype=np.float32)
        crop = np.array(crop)
        if crop.size == 0:
            return image, offset
        nxs, nys = crop[0], crop[1]

        def do_crop(im, left_x, top_y, margin=0.0):
            if len(im.shape)> 2:
                y, x = (im.shape[1], im.shape[2])
            else:
                y, x = im.shape
            x -= margin
            y -= margin

            left_x = int(max(0, min(left_x, x - nxs)))
            top_y = int(max(0, min(top_y, y - nys)))
            right_x = left_x + nxs
            bot_y = top_y + nys

            if (left_x > x or right_x < margin or right_x > x or
                top_y < margin or top_y > y or bot_y < margin or bot_y > y):
                print('negative', left_x, right_x, top_y, bot_y, 'constr', x,y)
                if len(im.shape) > 2:
                    return im[: , 0:nys, 0:nxs], (0,0)
                else:
                    return im[0:nys, 0:nxs], (0,0)
            if len(im.shape) > 2:
                return im[:, top_y:bot_y, left_x:right_x], (left_x, top_y)
            else:
                return im[top_y:bot_y, left_x:right_x], (left_x, top_y)

        centroid = key_pts[-1]
        #print(centroid)
        #centroid += np.random.uniform(low=-dither, high=dither, size=(2))
        off_x = int(centroid[0] - nxs / 2)
        off_y = int(centroid[1] - nys / 2)

        image, (off_x, off_y) = do_crop(image, off_x, off_y)
        offset = np.array([off_x, off_y], dtype=np.float32)
        return image, offset