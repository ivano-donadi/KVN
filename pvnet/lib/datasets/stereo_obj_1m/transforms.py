##########################
# Taken as-is from https://github.com/xingyul/stereobj-1m/blob/master/data_loader/transforms.py
##########################

import numpy as np
import random
import torchvision
import cv2
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, R=None, t=None, K=None):
        for trans in self.transforms:
            img, R, t, K = \
                trans(img, R, t, K)
        return img, R, t, K

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, img, R, t, K):
        img = np.asarray(img).astype(np.float32) / 255.
        R = np.asarray(R).astype(np.float32)
        t = np.asarray(t).astype(np.float32)
        K = np.asarray(K).astype(np.float32)
        return img, R, t, K


class Normalize(object):
    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, img, R, t, K):
        img -= self.mean
        img /= self.std
        img = img.astype(np.float32)
        return img, R, t, K


class ColorJitter(object):
    def __init__(
        self,
        brightness=None,
        contrast=None,
        saturation=None,
        hue=None,
    ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, R, t, K):
        image = np.asarray(
            self.color_jitter(
                Image.fromarray(np.ascontiguousarray(image, np.uint8))))
        return image, R, t, K


class RandomRotation(object):
    def __init__(self, degrees=360):
        self.random_rotation = torchvision.transforms.RandomRotation(
            degrees=degrees
        )

    def __call__(self, image, R, t, K):
        image = np.asarray(
            self.random_rotation(
                Image.fromarray(np.ascontiguousarray(image, np.uint8))))
        return image, R, t, K


class RandomBlur(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, R, t, K):
        if random.random() < self.prob:
            sigma = np.random.choice([3, 5, 7, 9])
            image = cv2.GaussianBlur(image, (sigma, sigma), 0)
        return image, R, t, K

import os
class RandomBackground(object):

    def __init__(self, directory, prob = 0.5):
        if directory :
          self.pths = [os.path.join(directory, pth) for pth in os.listdir(directory) ]
          self.prob = prob
        else :
          self.pths = []
          self.prob = prob
            

    def __call__(self, image, R, mask, K):
        if self.pths and random.random()<self.prob:
          bck_img = None
          while bck_img is None:
            bck_pth = random.sample(self.pths, 1)
            bck_img = cv2.imread(bck_pth[0])
          #bck_img = np.asarray(Image.open(bck_pth[0]));
          if bck_img.shape[2] == 1 :
            bck_img = cv2.cvtColor(bck_img, cv2.COLOR_GRAY2BGR )
          elif bck_img.shape[2] == 4 :
            bck_img = cv2.cvtColor(bck_img, cv2.COLOR_BGRA2BGR )
            
          size = (int(image.shape[1]), int(image.shape[0]));
          bck_img = cv2.resize( bck_img, size);
          fg = cv2.bitwise_and(image,image, mask=mask)
          i_mask = abs(1-mask)
          bg = cv2.bitwise_and(bck_img,bck_img, mask=i_mask)
          image = cv2.add(fg,bg)
          #cv2.imshow('image',image)
          #cv2.waitKey(1000)

        return image, R, mask, K



def make_transforms(is_train,cfg):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    print(cfg.train.bg_prob)
    if is_train is True:
        transform = Compose([
            RandomBackground(prob=cfg.train.bg_prob,directory="/home/donadi/Desktop/pvnet/data_ar/textures"),
            RandomBlur(0.2),
            ColorJitter(0.1, 0.1, 0.05, 0.05),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
    else:
        transform = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    return transform
