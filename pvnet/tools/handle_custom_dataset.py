import os
from plyfile import PlyData
import numpy as np
from lib.csrc.fps import fps_utils
from lib.utils.linemod.opengl_renderer import OpenGLRenderer
import tqdm
from PIL import Image
from lib.utils import base_utils
import json
import cv2 as cv


def read_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    points = np.stack([data['x'], data['y'], data['z']], axis=1)
    return points


def sample_fps_points(data_root):
    ply_path = os.path.join(data_root, 'model.ply')
    ply_points = read_ply_points(ply_path)
    fps_points = fps_utils.farthest_point_sampling(ply_points, 8, True)
    np.savetxt(os.path.join(data_root, 'fps.txt'), fps_points)


def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def record_ann(model_meta, img_id, ann_id, images, annotations):
    data_root = model_meta['data_root']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    pose_dir = os.path.join(data_root, 'pose')
    rgb_dir = os.path.join(data_root, 'rgb')
    mask_dir = os.path.join(data_root, 'mask')

    inds = range(len(os.listdir(rgb_dir)))

    for ind in tqdm.tqdm(inds):
        rgb_path = os.path.join(rgb_dir, '{}.jpg'.format(ind))

        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        pose_path = os.path.join(pose_dir, 'pose{}.npy'.format(ind))
        pose = np.load(pose_path)
        corner_2d = base_utils.project(corner_3d, K, pose)
        center_2d = base_utils.project(center_3d[None], K, pose)[0]
        fps_2d = base_utils.project(fps_3d, K, pose)

        mask_path = os.path.join(mask_dir, '{}.png'.format(ind))

        ann_id += 1
        anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})
        anno.update({'data_root': rgb_dir})
        anno.update({'type': 'real', 'cls': 'cat'})
        annotations.append(anno)

    return img_id, ann_id

def record_ann_stereo(model_meta, img_id, ann_id, images, annotations, baseline):
    data_root = model_meta['data_root']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    pose_dir = os.path.join(data_root, 'pose')
    rgb_dir = os.path.join(data_root, 'rgb')
    mask_dir = os.path.join(data_root, 'mask')

    inds = range(len(os.listdir(rgb_dir)) // 2)



    for ind in tqdm.tqdm(inds):
        rgb_path_L = os.path.join(rgb_dir, '{}_L.jpg'.format(ind))
        rgb_path_R = os.path.join(rgb_dir, '{}_R.jpg'.format(ind))

        rgb_L = Image.open(rgb_path_L)
        img_size = rgb_L.size
        img_id += 1
        info = {'file_name_L': rgb_path_L, 'file_name_R': rgb_path_R,'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        pose_path_L = os.path.join(pose_dir, 'pose{}_L.npy'.format(ind))
        pose_path_R = os.path.join(pose_dir, 'pose{}_R.npy'.format(ind))
        pose_L = np.load(pose_path_L)
        pose_R = np.load(pose_path_R)
        corner_2d_L = base_utils.project(corner_3d, K, pose_L)
        corner_2d_R = base_utils.project(corner_3d, K, pose_R)
        center_2d_L = base_utils.project(center_3d[None], K, pose_L)[0]
        center_2d_R = base_utils.project(center_3d[None], K, pose_R)[0]
        fps_2d_L = base_utils.project(fps_3d, K, pose_L)
        fps_2d_R = base_utils.project(fps_3d, K, pose_R)

        mask_path_L = os.path.join(mask_dir, '{}_L.png'.format(ind))
        mask_path_R = os.path.join(mask_dir, '{}_R.png'.format(ind))

        ann_id += 1
        anno = {'mask_path_L': mask_path_L,'mask_path_R': mask_path_R, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d_L': corner_3d.tolist(),'corner_3d_R': corner_3d.tolist(), 'corner_2d_L': corner_2d_L.tolist(), 'corner_2d_R': corner_2d_R.tolist()})
        anno.update({'center_3d_L': center_3d.tolist(), 'center_3d_R': center_3d.tolist(), 'center_2d_L': center_2d_L.tolist(), 'center_2d_R': center_2d_R.tolist()})
        anno.update({'fps_3d_L': fps_3d.tolist(),'fps_3d_R': fps_3d.tolist(), 'fps_2d_L': fps_2d_L.tolist(), 'fps_2d_R': fps_2d_R.tolist()})
        anno.update({'K': K.tolist(), 'pose_L': pose_L.tolist(), 'pose_R': pose_R.tolist()})
        anno.update({'data_root': rgb_dir})
        anno.update({'baseline': baseline})
        anno.update({'type': 'real', 'cls': 'cat'})
        annotations.append(anno)

    return img_id, ann_id

def write_inference_meta(data_root, K, corner_3d, center_3d, fps_3d, stereo = False, baseline = 0.):
    s = cv.FileStorage(os.path.join(data_root, 'inference_meta.yaml'), cv.FileStorage_WRITE)
    s.write('K', K)
    s.write('corner_3d', corner_3d)
    s.write('center_3d', center_3d)
    s.write('fps_3d', fps_3d)
    if stereo:
        s.write('baseline', baseline)
    s.release()

def read_inference_meta( inference_meta_fn ):
    s = cv.FileStorage(inference_meta_fn, cv.FileStorage_READ)
    K = s.getNode('K').mat()
    corner_3d = s.getNode('corner_3d').mat()
    center_3d = s.getNode('center_3d').mat()
    fps_3d = s.getNode('fps_3d').mat()
    s.release()
    return K, corner_3d, center_3d, fps_3d

def read_inference_meta_stereo( inference_meta_fn ):
    s = cv.FileStorage(inference_meta_fn, cv.FileStorage_READ)
    K = s.getNode('K').mat()
    baseline = s.getNode('baseline').real()
    corner_3d = s.getNode('corner_3d').mat()
    center_3d = s.getNode('center_3d').mat()
    fps_3d = s.getNode('fps_3d').mat()
    s.release()
    return K, baseline, corner_3d, center_3d, fps_3d


def custom_to_coco(data_root, stereo=False, baseline = 0.):
    model_path = os.path.join(data_root, 'model.ply')

    renderer = OpenGLRenderer(model_path)
    K = np.loadtxt(os.path.join(data_root, 'camera.txt'))

    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.loadtxt(os.path.join(data_root, 'fps.txt'))

    model_meta = {
        'K': K,
        'corner_3d': corner_3d,
        'center_3d': center_3d,
        'fps_3d': fps_3d,
        'data_root': data_root,
    }

    write_inference_meta(data_root, K, corner_3d, center_3d, fps_3d, stereo, baseline)
    
    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    if stereo:
        img_id, ann_id = record_ann_stereo(model_meta, img_id, ann_id, images, annotations, baseline)
    else:
        img_id, ann_id = record_ann(model_meta, img_id, ann_id, images, annotations)
      
    categories = [{'supercategory': 'none', 'id': 1, 'name': 'cat'}]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    anno_path = os.path.join(data_root, 'train.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)


def prepare_dataset(data_root):
  data_root = os.path.expanduser(data_root)
  sample_fps_points(data_root)
  custom_to_coco(data_root, False)

def prepare_dataset_stereo(data_root, baseline):
  data_root = os.path.expanduser(data_root)
  sample_fps_points(data_root)
  custom_to_coco(data_root, True, baseline)