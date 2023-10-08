import numpy as np
import lib.utils.pvnet.pvnet_pose_utils as pvnet_pose_utils
from scipy import spatial
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

auc_step = 0.005

def kps_for_obj(obj_name):
    if obj_name == "heart_0":
        return np.asarray([[4.169e-3,2.1632e-2,-9.52e-4],[-7.708e-3,-3.2392e-2,-7.17e-4],[0.034255 ,-0.000747, 0.000808],[-0.031651, 0.012909, -0.000851]])
    elif obj_name == "tree_0":
        return np.asarray([[-0.002898, 0.01193, -0.066737],[0.002179, -0.0066989, 0.042574],[0.011587, -0.047003, 0.016152],[-.0006609, 0.047003, 0.016152]])
    elif obj_name == "mug_0":
        return np.asarray([[0.0, 0.0, 0.052], [0.0, 0.0, -0.03], [-0.002, 0.058, 0.04]])
    elif obj_name == "mug_1":
        return np.asarray([[0.0, 0.0, 0.064],[0.0, 0.0, -0.02499],[-0.001, 0.056, 0.057]])
    elif obj_name == "mug_2":
        return np.asarray([[-0.004, 0.04, -0.002],[-0.002, -0.044999, -0.002],[0.057, 0.0249999, 0.002]])
    elif obj_name == "mug_3":
        return np.asarray([[-0.004999, -0.001, 0.055],[-0.004999, -0.001, -0.047999],[0.0559359, 0.029091, 0.043864]])
    elif obj_name == "mug_4":
        return np.asarray([[0.0, -0.001982, 0.068039], [-0.0015419, -0.003598, -0.056], [0.0170209, 0.071298, 0.044347]])
    elif obj_name == "mug_5":
        return np.asarray([[-0.0019149, -0.002614, 0.092042],[-0.0026879, -0.000142, -0.049397],[0.0604299, -0.010729, 0.072368]])
    elif obj_name == "mug_6":
        return np.asarray([[-0.000197, -0.00168, 0.0960169],[0.0004909, -0.002726, -0.056566],[-0.002965, 0.058606, 0.07328899]])
    elif obj_name == "ball_0":
        return np.asarray([[0.0, 0.0, 0.0]])
    elif obj_name == "bottle_0":
        return np.asarray([[0.0, 0.0, 0.04799],[0.0, 0.0, -0.04]])
    elif obj_name == "bottle_1":
        return np.asarray([[0.0, 0.08, 0.0],[0.0, -0.073, 0.0]])
    elif obj_name == "bottle_2":
        return np.asarray([[0.0, 0.093999, 0.0],[0.0, -0.078, 0.0]])
    elif obj_name == "cup_0":
        return np.asarray([[0.0, 0.0, 0.031],[0.0, 0.0, -0.038]])
    elif obj_name == "cup_1":
        return np.asarray([[0.0, 0.0, 0.062],[0.0, 0.0, -0.08399]])
    else:
        print('Warn: no keypoints for unknown object name:', obj_name )
        return None

def cm_degree_5_metric(pose_pred, pose_targets):
    translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
    rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
    return (translation_distance < 5 and angular_distance < 5)

def projection_2d(model, pose_pred, pose_targets, K, threshold=5):
    model_2d_pred = pvnet_pose_utils.project(model, K, pose_pred)
    model_2d_targets = pvnet_pose_utils.project(model, K, pose_targets)
    proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
    return (proj_mean_diff < threshold)

def compute_mean_dist(obj_model, pose_pred, pose_targets, syn):
    model_pred = np.dot(obj_model, pose_pred[:, :3].T) + pose_pred[:,3]
    model_targets = np.dot(obj_model, pose_targets[:, :3].T) + pose_targets[:, 3]
    if syn:
        mean_dist_index = spatial.cKDTree(model_pred)
        mean_dist, _ = mean_dist_index.query(model_targets, k=1)
        mean_dist = np.mean(mean_dist)
    else:
        mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
    return mean_dist

def add_metric(obj_model, obj_diameter, pose_pred, pose_targets, syn, percentage=0.1):
    diameter = obj_diameter * percentage
    mean_dist = compute_mean_dist(obj_model, pose_pred, pose_targets, syn)
    return mean_dist, mean_dist < diameter

def it_pnp_metrics(obj_model, obj_diameter, K, baseline, img_height, img_width, kpt_2d_L, kpt_2d_R, kpt_3d, var_L, var_R, pose_targets, icp=False, syn=False, percentage=0.1, initial_guess=None, obj_name = None):
    diameter = obj_diameter * percentage

    if var_L is not None:
        try:
            var_L = np.linalg.inv(var_L)
        except:
            var_L = var_L
            var_R = np.full(var_R.shape,np.eye(var_R.shape[1]))
        try:
            var_R = np.linalg.inv(var_R)
        except:
            var_R = var_R
            var_L = np.full(var_L.shape,np.eye(var_L.shape[1]))
        use_variance = True
    else:
        use_variance = False

    r_mat, transl = pvnet_pose_utils.iterative_pnp(K, baseline, img_width, img_height, kpt_2d_L, kpt_2d_R, var_L, var_R, kpt_3d, initial_guess, use_variance)

    return pose_metrics(r_mat, transl, pose_targets, obj_model, diameter, syn, K, obj_name)

def pose_metrics(r_mat, transl, gt_pose, obj_model, diameter, syn, K, obj_name = None):

    #if obj_name is not None:
    #    tod_model = kps_for_obj(obj_name)
    #    if tod_model is not None:
    #        obj_model = tod_model

    pose_pred = np.c_[r_mat, transl]
    mean_dist = compute_mean_dist(obj_model, pose_pred, gt_pose, syn)

    mae = mean_dist
    cm2 = mean_dist < 0.02
    cmd5 = cm_degree_5_metric(pose_pred, gt_pose)
    add = (mean_dist < diameter)
    proj2d = projection_2d(obj_model, pose_pred, gt_pose, K)
    return (mae, cm2, cmd5, add, proj2d, pose_pred)

def mask_iou(output, batch, seg_name = 'aux_mask_L', mask_name = 'mask_L'):
    #mask_pred = torch.argmax(output[seg_name], dim=1)[0].detach().cpu().numpy()
    mask_pred = output[seg_name].cpu().numpy()
    mask_gt = batch[mask_name][0].detach().cpu().numpy().astype(np.int64)
    iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
    return (iou > 0.7)


def curve_from_mae(mae):
    n_steps = 100
    # range from 0 to 10 cm = 0.1 m
    step = 1e-1 / n_steps
    thresholds = [x * step for x in range(1,n_steps)]
    curve = []
    for t in thresholds:
        curve_val = np.count_nonzero(np.array(mae) < t)/len(mae)
        curve.append(curve_val)
    #curve = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[], []]
    #for dist in mae:
    #    for step in range(21):
    #        curr_thresh = step * auc_step
    #        curve[step].append(dist < curr_thresh)
    #return curve
    return curve, step

def auc_from_curve(curve, step):
    auc = 0.
    base = 1/len(curve)
    for val in curve:
        # area = step * height
        auc = auc + val * base
    return auc

def initial_guess_from_variance(var, kpt_2d, kpt_3d, K):
    traces = list(enumerate(np.trace(var, axis1 = 1, axis2 = 2)))
    traces.sort(key=lambda x: x[1])
    kpt_2d_from_idx = lambda x: kpt_2d[traces[x][0]]
    kpt_3d_from_idx = lambda x: kpt_3d[traces[x][0]]
    filtered_2d_kps = np.asarray([kpt_2d_from_idx(x) for x in range(4)])
    filtered_3d_kps = np.asarray([kpt_3d_from_idx(x) for x in range(4)])
    initial_guess = pvnet_pose_utils.pnp(filtered_3d_kps, filtered_2d_kps, K, method = cv2.SOLVEPNP_EPNP)
    cost = sum([traces[x][1] for x in range(4)])
    if cost == 0:
        cost = np.inf
    return initial_guess, cost

def plot_axes(ax, center_3d, transf, K, gt = False, length = 0.05):
    x_axis_end = center_3d + length*transf[:,0]
    y_axis_end = center_3d + length*transf[:,1]
    z_axis_end = center_3d + length*transf[:,2]
    x_axis_end_2d = pvnet_pose_utils.project(np.asarray([x_axis_end]) ,K, transf)
    y_axis_end_2d = pvnet_pose_utils.project(np.asarray([y_axis_end]) ,K, transf)
    z_axis_end_2d = pvnet_pose_utils.project(np.asarray([z_axis_end]) ,K, transf)
    center_2d = pvnet_pose_utils.project(np.asarray([center_3d]) ,K, transf)

    if gt:
        base_color = 'g'
        x_color = 'r'
        y_color = 'g'
        z_color = 'b'

    else:
        base_color = 'b'
        x_color = 'm'
        y_color = 'k'
        z_color = 'c'

    ax.plot(center_2d[0][0], center_2d[0][1], marker = mpl.markers.MarkerStyle(marker='x',fillstyle='full'), color= base_color)
    ax.plot([center_2d[0][0],x_axis_end_2d[0][0]], [center_2d[0][1],x_axis_end_2d[0][1]], color=x_color)
    ax.plot(x_axis_end_2d[0][0], x_axis_end_2d[0][1], marker = mpl.markers.MarkerStyle(marker='8',fillstyle='none'), color= x_color)
    ax.plot([center_2d[0][0],y_axis_end_2d[0][0]], [center_2d[0][1],y_axis_end_2d[0][1]], color=y_color)
    ax.plot(y_axis_end_2d[0][0], y_axis_end_2d[0][1], marker = mpl.markers.MarkerStyle(marker='8',fillstyle='none'), color= y_color)
    ax.plot([center_2d[0][0],z_axis_end_2d[0][0]], [center_2d[0][1],z_axis_end_2d[0][1]], color=z_color)
    ax.plot(z_axis_end_2d[0][0], z_axis_end_2d[0][1], marker = mpl.markers.MarkerStyle(marker='8',fillstyle='none'), color= z_color)


def plot_parallel_kps(img_L, img_R, output_L, output_R, K, kpt_3d, pose_pred, pose_gt_L, pose_gt_R, baseline, offset_L, offset_R):
    _, (ax1, ax2) = plt.subplots(1, 2)

    kpt_2d = pvnet_pose_utils.project(kpt_3d, K, pose_pred)
    kpt_2d_gt = pvnet_pose_utils.project(kpt_3d, K, pose_gt_L)
    #print('gt',kpt_2d_gt)
    kpt_2d_L = output_L['kpt_2d'][0].detach().cpu().numpy() 
    #print('bf',kpt_2d_L)
    kpt_2d_L += offset_L
    #print('af', kpt_2d_L)
    mask = np.zeros((720,1280), dtype=np.uint8)
    mask_pred = torch.argmax(output_L['seg'],dim=1)[0].detach().cpu().numpy()
    mask[int(offset_L[0,1]):int(offset_L[0,1])+126, int(offset_L[0,0]):int(offset_L[0,0])+224] = mask_pred
    plot_si_kps(ax1, output_L, img_L, mask, kpt_2d_gt, kpt_2d, kpt_2d_L)

    mask = np.zeros((720,1280), dtype=np.uint8)
    mask_pred = torch.argmax(output_R['seg'],dim=1)[0].detach().cpu().numpy()
    mask[int(offset_R[0,1]):int(offset_R[0,1])+126, int(offset_R[0,0]):int(offset_R[0,0])+224] = mask_pred
    pose_pred[0,3]-=baseline
    kpt_2d = pvnet_pose_utils.project(kpt_3d, K, pose_pred)
    kpt_2d_gt = pvnet_pose_utils.project(kpt_3d, K, pose_gt_R)
    kpt_2d_R = output_R['kpt_2d'][0].detach().cpu().numpy() + offset_R
    plot_si_kps(ax2, output_R, img_R, mask, kpt_2d_gt, kpt_2d, kpt_2d_R)

    plt.title(img_L)
    plt.show()

def plot_stereo_kps(img_L, img_R, output, K, kpt_3d, pose_pred, pose_gt, baseline):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    kpt_2d = pvnet_pose_utils.project(kpt_3d, K, pose_pred)
    kpt_2d_gt = pvnet_pose_utils.project(kpt_3d, K, pose_gt)
    kpt_2d_L = output['kpt_2d_L'][0].detach().cpu().numpy()
    mask_pred = torch.argmax(output['seg_L'],dim=1)[0].detach().cpu().numpy()
    plot_si_kps(ax1, output, img_L, mask_pred, kpt_2d_gt, kpt_2d, kpt_2d_L)

    mask_pred = torch.argmax(output['seg_R'],dim=1)[0].detach().cpu().numpy()
    pose_pred[0,3]-=baseline
    pose_gt[0,3]-=baseline
    kpt_2d = pvnet_pose_utils.project(kpt_3d, K, pose_pred)
    kpt_2d_gt = pvnet_pose_utils.project(kpt_3d, K, pose_gt)
    kpt_2d_R = output['kpt_2d_R'][0].detach().cpu().numpy()
    plot_si_kps(ax2, output, img_R, mask_pred, kpt_2d_gt, kpt_2d, kpt_2d_R)
    print(img_L)
 #   plt.title(img_L)
    plt.show()


def plot_si_kps(ax, output, img, mask, gt_kps, pred_kps, net_kps):
    img = Image.open(img)
    ax.imshow(img)
    ax.imshow(mask, alpha=0.25)

    for kpg, kpp, kpn in zip(gt_kps, pred_kps, net_kps):
        ax.plot(kpg[0],kpg[1], marker='x', color='g')
        ax.plot(kpp[0],kpp[1], marker='x', color='m')
#        ax.plot([kpg[0], kpp[0]],[kpg[1],kpp[1]], color = 'r')
        ax.plot(kpn[0],kpn[1], marker='x', color='b')


