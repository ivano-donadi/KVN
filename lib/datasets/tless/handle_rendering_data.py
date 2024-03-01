import os
import glob
from PIL import Image
import numpy as np
import tqdm
from lib.utils.renderer import render_utils
from lib.utils.base_utils import read_pickle
import json
import cv2


blender = '/home/pengsida/Software/blender-2.79a-linux-glibc219-x86_64/blender'
blank_blend = 'lib/datasets/tless/blank.blend'
backend = 'lib/datasets/tless/render_backend.py'

data_root = 'data/tless'
ply_path_pattern = os.path.join(data_root, 'models_cad/colobj_{:02}.ply')
output_dir_pattern = os.path.join(data_root, 'renders/{}')


def get_bg_imgs():
    bg_path = 'data/tless/bg_imgs.npy'
    bg_dataset = 'data/sun/JPEGImages'
    if os.path.exists(bg_path):
        return

    img_paths = glob.glob(os.path.join(bg_dataset, '*.jpg'))
    bg_imgs = []

    for img_path in tqdm.tqdm(img_paths):
        img = Image.open(img_path)
        row, col = img.size
        if row > 500 and col > 500:
            bg_imgs.append(img_path)

    np.save(bg_path, bg_imgs)


def get_poses(num_samples):
    euler = render_utils.ViewpointSampler.sample_sphere(num_samples)
    x = np.random.uniform(-0.01, 0.01, num_samples)
    y = np.random.uniform(-0.01, 0.01, num_samples)
    z = np.random.uniform(0.18, 0.2, num_samples)
    # z = np.random.uniform(0.27, 0.30, num_samples)
    translation = np.stack([x, y, z], axis=1)
    poses = np.concatenate([euler, translation], axis=1)
    np.save('data/tless/poses.npy', poses)


def _render(obj_id, num_syn):
    ply_path = ply_path_pattern.format(obj_id)
    output_dir = output_dir_pattern.format(obj_id)
    os.system('{} {} --background --python {} -- --cad_path {} --output_dir {} --num_syn {}'.format(blender, blank_blend, backend, ply_path, output_dir, num_syn))

    depth_paths = glob.glob(os.path.join(output_dir, '*.exr'))
    for depth_path in depth_paths:
        render_utils.exr_to_png(depth_path)


def render():
    get_bg_imgs()
    num_syn = 5000
    get_poses(num_syn)
    obj_ids = [30]
    for obj_id in obj_ids:
        _render(obj_id, num_syn)


def render_to_coco():
    data_root = 'data/tless/renders/'
    obj_ids = [i + 1 for i in range(30)]

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    for obj_id in tqdm.tqdm(obj_ids):
        obj_dir = os.path.join(data_root, str(obj_id))
        pkl_paths = glob.glob(os.path.join(obj_dir, '*.pkl'))
        for pkl_path in tqdm.tqdm(pkl_paths):
            rgb_path = pkl_path.replace('_RT.pkl', '.png')
            mask_path = pkl_path.replace('_RT.pkl', '_depth.png')

            if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
                continue

            rgb = Image.open(rgb_path)
            img_size = rgb.size
            img_id += 1
            info = {'rgb_path': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
            images.append(info)

            K_P = read_pickle(pkl_path)

            ann_id += 1
            anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': obj_id, 'id': ann_id}
            anno.update({'K': K_P['K'].tolist(), 'pose': K_P['RT'].tolist()})
            annotations.append(anno)

    categories = [{'supercategory': 'none', 'id': obj_id, 'name': str(obj_id)} for obj_id in obj_ids]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}
    anno_path = os.path.join(data_root, 'render.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)
