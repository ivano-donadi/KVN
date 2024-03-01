import matplotlib.pyplot as plt
from lib.utils import img_utils
from lib.utils.pvnet import pvnet_config, pvnet_pose_utils
import numpy as np
import torch
import matplotlib.patches as patches

mean = pvnet_config.mean
std = pvnet_config.std

kp_colors = ['#dcbeff', '#9A6324', '#808000', '#469990', '#FFFFFF', '#e6194B', '#f58231', '#42d4f4', '#f032e6']


def visualize_ann(img, kpt_2d, mask, savefig=False):
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.plot(kpt_2d[:, 0], kpt_2d[:, 1], '.')
    ax2.imshow(mask)
    if savefig:
        plt.savefig('test.jpg')
    else:
        plt.show()


def visualize_linemod_ann(img, kpt_2d, mask, savefig=False):
    img = img_utils.unnormalize_img(img, mean, std, False).permute(1, 2, 0)
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    if kpt_2d is not None:
        ax1.plot(kpt_2d[:, 0], kpt_2d[:, 1], '.')
    ax2.imshow(mask)
    plt.show()

def visualize_tod_ann(img_L, img_R, kpt_2d_L, kpt_2d_R, mask_L, mask_R, savefig=False):
    img_L = img_utils.unnormalize_img(img_L, mean, std, False).permute(1, 2, 0)
    img_R = img_utils.unnormalize_img(img_R, mean, std, False).permute(1, 2, 0)
    _, ((ax00, ax01),(ax10,ax11)) = plt.subplots(2, 2)
    ax00.set_title('GT keypoints')
    ax00.imshow(img_L)
    if kpt_2d_L is not None:
        for i in range(kpt_2d_L.shape[0]):
            ax00.plot(kpt_2d_L[i, 0], kpt_2d_L[i, 1], 'o', color=kp_colors[i], markeredgecolor='k', markersize=5)
    ax01.set_title('GT mask')
    ax01.imshow(mask_L)

    ax10.set_title('Predicted keypoints')
    ax10.imshow(img_R)
    if kpt_2d_R is not None:
        for i in range(kpt_2d_R.shape[0]):
            ax10.plot(kpt_2d_R[i, 0], kpt_2d_R[i, 1], 'o', color=kp_colors[i], markeredgecolor='k', markersize=5)
    ax11.set_title('Predicted mask')
    ax11.imshow(mask_R)
    plt.show()

def visualize_hm_ann(img, kpt_2d, mask, hm):
    img = img_utils.unnormalize_img(img, mean, std, False).permute(1, 2, 0)
    _, ((ax11,ax12), (ax21, ax22)) = plt.subplots(2, 2)
    ax11.imshow(img)
    ax12.imshow(mask)
    ax21.imshow(img)
    if kpt_2d is not None:
        ax21.plot(kpt_2d[:, 0], kpt_2d[:, 1], 'x')
    ax22.imshow(hm)
    plt.show()

def visualize_dsac_results(img, hps, probs, gt_kps = None):
    img = img_utils.unnormalize_img(img, mean, std, False).permute(1, 2, 0)
    h, w, c = img.shape
    bins = np.zeros((h,w), dtype = np.float32)
    rn, vn, _ = hps.shape
    for r in range(rn):
        for k in range(vn):
            x = int(hps[r,k,0])
            y = int(hps[r,k,1])
            val = max(0,np.log(1+probs[r,k].item()))
            if x >= 0 and x < w and y >=0 and y < h:
                bins[y,x] += val
    #max_val = np.max(bins)
    #min_val = np.min(bins)
    #bins = (bins - min_val)/(max_val-min_val)
    #bins = np.log(2+bins)
    _, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(img)
    ax2.imshow(bins)
    if gt_kps is not None:
        for k in range(vn):
            avg_kp = (hps[:,k] * probs[:,k,None]).sum(0).detach().cpu().numpy()
            ax1.plot(gt_kps[k,0], gt_kps[k,1], marker='o',markeredgecolor='k', markerfacecolor=kp_colors[k], markersize='5')
            ax2.plot(gt_kps[k,0], gt_kps[k,1], marker='o',markeredgecolor=kp_colors[k], markerfacecolor='none', markersize='6', linewidth = 0.1)
            #ax2.plot(avg_kp[0], avg_kp[1], marker='x',color=kp_colors[k], markersize='5')
    plt.show()

def visualize_dsac_results_split(img, hps, probs, gt_kps=None):
    img = img_utils.unnormalize_img(img, mean, std, False).permute(1, 2, 0)
    h, w, c = img.shape
    rn, vn, _ = hps.shape
    bins = np.zeros((vn,h,w), dtype = np.float32)
    for r in range(rn):
        for k in range(vn):
            x = int(hps[r,k,0])
            y = int(hps[r,k,1])
            val = max(0,probs[r,k])
            if x >= 0 and x < w and y >=0 and y < h:
                bins[k,y,x] += val
    for k in range(vn):
        max_val = np.max(bins[k])
        min_val = np.min(bins[k])
        bins[k] = (bins[k] - min_val)/(max_val-min_val)
    tot_imgs = vn + 1 
    cols = 2
    rows = int(np.ceil(tot_imgs/cols))
    _, axs = plt.subplots(rows, cols)
    axs[0][0].imshow(img)
    for i in range(1,tot_imgs):
        row = int(i // cols)
        col = int(i % cols)
        axs[row][col].imshow(bins[i-1])
        avg_kp = (hps[:,i-1] * probs[:,i-1,None]).sum(0).detach().cpu().numpy()
        if gt_kps is not None:
            axs[0][0].plot(gt_kps[i-1,0], gt_kps[i-1,1], marker='o',markeredgecolor=kp_colors[i-1], markerfacecolor='none', markersize='5')
            axs[0][0].plot(avg_kp[0], avg_kp[1], marker='x',color=kp_colors[i-1], markersize='5')
            axs[row][col].plot(gt_kps[i-1,0], gt_kps[i-1,1], marker='o',markeredgecolor=kp_colors[i-1], markerfacecolor='none', markersize='5')
            axs[row][col].plot(avg_kp[0], avg_kp[1], marker='x',color=kp_colors[i-1], markersize='5')
            axs[row][col].plot([avg_kp[0],gt_kps[i-1,0]], [avg_kp[1],gt_kps[i-1,1]], color=kp_colors[i-1])
    plt.show() 

def visualize_error_maps(batch_v, output_v, keypoints, mask):
    batch_v = batch_v.permute(0, 2, 3, 1)
    b, h, w, vn_2 = batch_v.shape
    vn = vn_2//2
    batch_v = batch_v.view(b, h, w, vn_2//2, 2)
    output_v = output_v.permute(0, 2, 3, 1)
    output_v = output_v.view(b, h, w, vn_2//2, 2)
    diff_v = (batch_v - output_v).pow(2).sum(4).sqrt().detach().cpu().numpy()
    tot_imgs = vn 
    cols = 2
    rows = int(np.ceil(tot_imgs/cols))
    _, axs = plt.subplots(rows, cols)
    keypoints = keypoints.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    for i in range(tot_imgs):
        errors = diff_v[0,:,:,i] * mask[0]
        max = errors.max()
        min = errors.min()
        errors = (errors-min)/(max-min)
        row = int(i // cols)
        col = int(i % cols)
        axs[row][col].imshow(errors)
        axs[row][col].plot(keypoints[0,i,0], keypoints[0,i,1], marker = 'o', color=kp_colors[i], markeredgecolor='k', markersize=5)
    plt.show()

def visualize_angle_error_maps(batch_v, output_v, keypoints, mask):
    batch_v = batch_v.permute(0, 2, 3, 1)
    b, h, w, vn_2 = batch_v.shape
    vn = vn_2//2
    batch_v = batch_v.view(b, h, w, vn_2//2, 2)
    output_v = output_v.permute(0, 2, 3, 1)
    output_v = output_v.view(b, h, w, vn_2//2, 2)
    #  both batch and output have norm = 1 so cos(t) = <batch,output>/(||batch||*||output||) = <batch,output>
    diff_v = (batch_v[:,:,:,:,0]*output_v[:,:,:,:,0] + batch_v[:,:,:,:,1]*output_v[:,:,:,:,1])
    tot_imgs = vn 
    cols = 2
    rows = int(np.ceil(tot_imgs/cols))
    _, axs = plt.subplots(rows, cols)
    keypoints = keypoints.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    for i in range(tot_imgs):
        errors = diff_v[0,:,:,i] * mask[0]
        max = errors.max()
        min = errors.min()
        errors = (errors-min)/(max-min)
        row = int(i // cols)
        col = int(i % cols)
        axs[row][col].imshow(errors)
        axs[row][col].plot(keypoints[0,i,0], keypoints[0,i,1], marker = 'o', color=kp_colors[i], markeredgecolor='k', markersize=5)
    plt.show()

def visualize_3d_bbox(img, corner_3d, K , pose_pred, pose_gt, offset):
    corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt) - offset
    corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred) - offset
    _, ax = plt.subplots(1)
    ax.imshow(img_utils.unnormalize_img(img, mean, std, False).permute(1, 2, 0))
    ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=2, edgecolor='b'))
    ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=2, edgecolor='b'))

    ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=2, edgecolor='g'))
    ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=2, edgecolor='g'))
    plt.show()