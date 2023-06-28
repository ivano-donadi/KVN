import numpy as np
import os
import random
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
from PIL import Image


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpts=None, mask=None):
        for t in self.transforms:
            img, kpts, mask = t(img, kpts, mask)
        return img, kpts, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):

    def __call__(self, img, kpts, mask):
        return np.asarray(img).astype(np.float32) / 255., kpts, mask


class Normalize(object):

    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, img, kpts, mask):
        img -= self.mean
        img /= self.std
        if self.to_bgr:
            img = img.transpose(2, 0, 1).astype(np.float32)
        return img, kpts, mask


class ColorJitter(object):

    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, kpts, mask):
        image = np.asarray(self.color_jitter(Image.fromarray(np.ascontiguousarray(image, np.uint8))))
        return image, kpts, mask


class RandomBlur(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, kpts, mask):
        if random.random() < self.prob:
            sigma = np.random.choice([3, 5, 7, 9])
            image = cv2.GaussianBlur(image, (sigma, sigma), 0)
        return image, kpts, mask


class RandomBackground(object):

    def __init__(self, directory, prob = 0.5):
        if directory :
          self.pths = [os.path.join(directory, pth) for pth in os.listdir(directory) ]
          self.prob = prob
        else :
          self.pths = []
          self.prob = prob
            

    def __call__(self, image, kpts, mask):
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

        return image, kpts, mask
        
def make_transforms(is_train, bkg_imgs_dir = "", bg_prob = 0.):
    if is_train is True:
        transform = Compose(
            [
                RandomBackground(bkg_imgs_dir, bg_prob),
                RandomBlur(0.5),
                ColorJitter(0.1, 0.1, 0.05, 0.1),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transform
