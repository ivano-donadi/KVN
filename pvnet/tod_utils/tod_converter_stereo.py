from __future__ import annotations
import sys
import os
import os.path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.handle_custom_dataset import read_inference_meta 
from lib.utils.pvnet import pvnet_pose_utils
from tod_utils import utils
import numpy as np
import argparse
from PIL import Image
import json
#mport OpenEXR as exr
#import Imath
from pathlib import Path
import random
import cv2
import shutil


import matplotlib.pyplot as plt
import matplotlib as mpl

count_mask_pixels = False

def plot_axes(ax, center_3d, transf, K, img, uvds, kps, gt = False, length = 0.05):
    x_axis_end = center_3d[0] + length*np.array([1., 0., 0.])#transf[:,0]
    y_axis_end = center_3d[0] + length*np.array([0., 1., 0.])#transf[:,1]
    z_axis_end = center_3d[0] + length*np.array([0., 0., 1.])#transf[:,2]
    x_axis_end_2d = pvnet_pose_utils.project(np.asarray([x_axis_end]) ,K, transf)
    y_axis_end_2d = pvnet_pose_utils.project(np.asarray([y_axis_end]) ,K, transf)
    z_axis_end_2d = pvnet_pose_utils.project(np.asarray([z_axis_end]) ,K, transf)
    center_2d = pvnet_pose_utils.project(center_3d ,K, transf)

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

    proj_kps = pvnet_pose_utils.project(kps ,K, transf)

    
    for i in range(proj_kps.shape[0]):
        ax.plot(proj_kps[i][0],proj_kps[i][1], marker = mpl.markers.MarkerStyle(marker='x',fillstyle='full'), color='w')

    for i in range(uvds.shape[0]):
        ax.plot(uvds[i][0],uvds[i][1], marker = mpl.markers.MarkerStyle(marker='x',fillstyle='full'), color='k')


    for i in range(uvds.shape[0]):
        ax.plot([uvds[i][0], proj_kps[i][0]],[uvds[i][1],proj_kps[i][1]], color='r')


def poses_for_texture(texture, model):
    if model == 'heart_0':
        if texture in [0,1,2,4,5,6,9]:
            return [0,1,2,3]
        elif texture == 3:
            return [0,1,3]
        elif texture == 7:
            return [0]
        elif texture == 8:
            return [0,1,2]
    elif model == 'tree_0':
        if texture in [0,1,9]:
            return [0,2,3]
        elif texture in [2,6,8]:
            return [1,2,3]
        elif texture in [3,5]:
            return [0,1,2,3]
        elif texture == 4:
            return [0,1,3]
        elif texture == 7:
            return [0]
    elif model == 'mug_0':
        if texture in [1,2,3,4,5,9]:
            return [0,1,2,3]
        elif texture in [0]:
            return [0,1,2]
        elif texture in [6]:
            return [1]
        elif texture in [7]:
            return [0,3]
        elif texture in [8]:
            return [0,1]
    elif model == 'mug_1':
        if texture in [0,1,5,9]:
            return [0,1,2,3]
        elif texture in [2,6,8]:
            return [0,1,2]
        elif texture in [3,4]:
            return [1,2]
        elif texture in [7]:
            return [1]
    elif model == 'mug_2':
        if texture in [0,1,2,3,4,5,6,7,9]:
            return [0,1,2,3]
        if texture in [8]:
            return [0,1,2]
    elif model == 'mug_3':
        if texture in [0,1,2,3,5,6,7,8,9]:
            return [0,1,2,3]
        if texture in [4]:
            return [0,1,2]
    elif model == 'mug_4':
        if texture in [0,2,3,5,6,7,8,9]:
            return [0,1,2,3]
        if texture in [4]:
            return [0,1,2]
        if texture in [1]:
            return [0,1,3]
    elif model == 'mug_5':
        if texture in [0,3,4,5,6,7,8,9]:
            return [0,1,2,3]
        if texture in [1]:
            return [1,2,3]
        if texture in [2]:
            return [0,1,3]
    elif model == 'mug_6':
        if texture in [0,8]:
            return [0,1,2]
        if texture in [1,2,3,4,5,6,9]:
            return [0,1,2,3]
        if texture in [7]:
            return [1,2,3]
    elif model == 'bottle_0':
        if texture in [0,1,2,3,4,5,6,8,9]:
            return [0,1,2,3]
        if texture in [7]:
            return [0,2,3]
    elif model == 'ball_0':
        if texture in [0,1,2,3,4,5,6,7,8,9]:
            return [0,1,2,3]
    elif model == 'bottle_1':
        if texture in [0,1,2,3,4,5,6,8,9]:
            return [0,1,2,3]
        if texture in [7]:
            return [0,1,3]
    elif model == 'bottle_2':
        if texture in [0,1,2,3,4,6,7,8,9]:
            return [0,1,2,3,4]
        if texture in [5]:
            return [1,2,3,4]
    elif model == 'cup_0':
        if texture in [0,1,2,5,7,8,9]:
            return [0,1,2,3]
        if texture in [3]:
            return [1,2,3]
        if texture in [4]:
            return [1,2]
        if texture in [6]:
            return [0,2,3]
    elif model == 'cup_1':
        if texture in [0,1,2,3,5,6,7,8,9]:
            return [0,1,2,3]
        if texture in [4]:
            return [0,2,3]

def read_depth_exr_file(filepath):
    exrfile = exr.InputFile(Path(filepath).as_posix())
    print('header',exrfile.header())
    raw_bytes = exrfile.channel('D')
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = np.reshape(depth_vector, (height, width))
    return depth_map

def process_image(folder_name,image_i,id, img_list, ann_list, class_id):
    base_file_name = '00000' if image_i<10 else '0000'
    base_file_name = base_file_name + str(image_i)
    mask_name = os.path.join(args.dataset_dir,folder_name,base_file_name + "_mask.png")
    if class_id == 'ball_0' and (('texture_3_pose_1' in folder_name and image_i == 30) or ('texture_1_pose_1' in folder_name and image_i == 31) or ('texture_0_pose_1' in folder_name and (image_i == 29 or image_i == 30))):
        print('Excluded empty mask image:', base_file_name)
        return
    
    is_symmetric = class_id in ['ball_0', 'bottle_0', 'bottle_1', 'cup_0', 'cup_1']

    curr_img_obj = {}
    curr_ann_obj = {}
    for camera in ['_L','_R']:
        file_name = base_file_name + camera
        img_name = os.path.join(args.dataset_dir,folder_name,file_name + '.png')
        pbtxt = os.path.join(args.dataset_dir,folder_name,file_name + '.pbtxt')
        try:
            img = Image.open(img_name,'r')
        except:
            print('img failed: ',img_name)
            return
        res = utils.read_contents_pb(pbtxt)
        cam = res[0]
        uvds= res[3]
        visible = np.sum(res[5])
        if not visible:
            print('Non visible kp: ',img_name)
            return
        p_matrix = utils.p_matrix_from_camera(cam)
        try:
            q_matrix = utils.q_matrix_from_camera(cam)
        except:
            print('img failed: ',img_name)
            return
        xyzs = utils.project_np(q_matrix, uvds.T)
        

        kps_t_mesh = utils.ortho_procrustes(obj.keypoints, xyzs.T[:, :3])

        true_trans = kps_t_mesh #np.asarray(res[6]).dot(transf)
        if is_symmetric:
            if class_id == 'bottle_1':
                a = true_trans[:3,1]
                rcol=0
            else:
                a = true_trans[:3,2]
                rcol = 0
            a_skew = np.array([[0., -a[2], a[1]], [a[2],0.,-a[0]], [-a[1], a[0], 0.]])
            a_2 = np.dot(a_skew, a_skew)
            theta, theta_2 = utils.orthogonal_x_rotation(a_skew, a_2, true_trans, rcol)
            rmat = utils.rot_mat_from_theta(theta, a_skew, a_2)
            aligned_rotation = np.dot(rmat, true_trans[:3,:3])

            if class_id == 'bottle_1':
                alt_condition = aligned_rotation[2,2] < 0
            else:
                alt_condition = aligned_rotation[2,1] > 0
            
            if alt_condition:
                rmat = utils.rot_mat_from_theta(theta_2, a_skew, a_2)
                aligned_rotation = np.dot(rmat, true_trans[:3,:3])

            if np.abs(aligned_rotation[2,0]) > 1e-10:
                raise RuntimeError("Symmetric object not correctly aligned")
            old_trans = true_trans.copy()
            true_trans[:3,:3]=aligned_rotation
            
        true_trans = true_trans[:3]
        fps_2d = pvnet_pose_utils.project(fps_3d, K, true_trans)
        corner_2d = pvnet_pose_utils.project(corner_3d, K, true_trans)
        center_2d = pvnet_pose_utils.project(center_3d, K, true_trans)

        if camera == '_L':
            curr_img_obj["file_name_L"] = img_name
            curr_img_obj["height"]= img.height
            curr_img_obj["width"]= img.width
            curr_img_obj["id"]= id

            # ball 0 object has no left mask annotation
            n_mask_path = os.path.join(args.dataset_dir,folder_name,base_file_name + "_mask.png")
            if not os.path.exists(n_mask_path):
                proj_kps = pvnet_pose_utils.project(obj.vertices,K,true_trans)
                n_mask, _ = obj.segmentation(proj_kps,(img.size[1], img.size[0]))
                cv2.imwrite(n_mask_path, n_mask)

            curr_ann_obj["mask_path_L"]= mask_name
            curr_ann_obj["image_id"]= id
            curr_ann_obj["category_id"]= 1
            curr_ann_obj["id"]= id
            curr_ann_obj["corner_3d_L"]= np.ndarray.tolist(corner_3d)
            curr_ann_obj["corner_2d_L"]= np.ndarray.tolist(corner_2d)
            curr_ann_obj["center_3d_L"]= np.ndarray.tolist(center_3d[0].reshape(3))
            curr_ann_obj["center_2d_L"]= np.ndarray.tolist(center_2d[0])
            curr_ann_obj["fps_3d_L"]= np.ndarray.tolist(fps_3d)
            curr_ann_obj["fps_2d_L"]= np.ndarray.tolist(fps_2d)
            curr_ann_obj["pose_L"]= np.ndarray.tolist(true_trans)
        else:
            
            # create right image mask
            n_mask_path = os.path.join(args.dataset_dir,folder_name,base_file_name + "_mask_R.png")
            
            if not os.path.exists(n_mask_path):
                proj_kps = pvnet_pose_utils.project(obj.vertices,K,true_trans)
                n_mask, _ = obj.segmentation(proj_kps,(img.size[1], img.size[0]))
                cv2.imwrite(n_mask_path, n_mask)

            curr_img_obj["file_name_R"]= img_name
            curr_ann_obj["mask_path_R"]= n_mask_path
            curr_ann_obj["corner_3d_R"]= np.ndarray.tolist(corner_3d)
            curr_ann_obj["corner_2d_R"]= np.ndarray.tolist(corner_2d)
            curr_ann_obj["center_3d_R"]= np.ndarray.tolist(center_3d[0].reshape(3))
            curr_ann_obj["center_2d_R"]= np.ndarray.tolist(center_2d[0])
            curr_ann_obj["fps_3d_R"]= np.ndarray.tolist(fps_3d)
            curr_ann_obj["fps_2d_R"]= np.ndarray.tolist(fps_2d)
            curr_ann_obj["pose_R"]= np.ndarray.tolist(true_trans)
            curr_ann_obj["K"]= np.ndarray.tolist(K)
            #TODO: extract from data
            curr_ann_obj["baseline"]=0.120007
            curr_ann_obj["data_root"]= folder_name
            curr_ann_obj["type"]= "real"
            curr_ann_obj["cls"]= class_id
            
            img_list.append(curr_img_obj)
            ann_list.append(curr_ann_obj)
        
        if  image_i == 1 and visualize and camera == '_R':
            
            if is_symmetric:
                _, (ax1, ax2) = plt.subplots(1,2)
                ax1.imshow(img)
                ax2.imshow(img)
                print('TT',old_trans[:3,:3])
                print('AT', true_trans[:3,:3])
                plot_axes(ax1,center_3d, old_trans[:3, :], K, img, fps_2d, fps_3d, gt = True, length = 0.05)
                plot_axes(ax2,center_3d, true_trans[:3, :], K, img, fps_2d, fps_3d, gt = False, length = 0.05)
                plt.show()
            else:
                _, ax = plt.subplots(1)
                proj_kps = pvnet_pose_utils.project(obj.keypoints[:,:3],K,true_trans)
                #for c in fps_2d :
                #    ax.plot(c[0],c[1], 'go',markersize=2)
                for c in proj_kps :
                    ax.plot(c[0],c[1], 'ro',markersize=1)
                
                ax.imshow(img)
                if camera == '_R':
                    n_mask = Image.open(n_mask_path,'r')
                    ax.imshow(n_mask, alpha=0.5)
            plt.show() 
            

######################################################################################
# MAIN
######################################################################################

parser = argparse.ArgumentParser(description='Tool for converting the  TOD dataset in the PVNET format',
                                     epilog="Will add all the images in DATASET_DIR as the training set, except for the ones belonging to the texture selected as test set")

parser.add_argument('-d', '--dataset_dir', 
                        help='Input directory containing the TOD dataset', required=True)    
parser.add_argument('-m', '--meta',  
                    help='metafiles folder', required=True)
parser.add_argument('-t',"--test_texture", 
                    help='id of the texture to keep as the test set (0-9)', 
                    default=9, type=int)
parser.add_argument('-v','--visualize', help='flag to activate visualization of transformations', default=False, type=bool)
parser.add_argument('-o', '--output_dir', help='directory in which to store the output (train.json,test.json,validation.json). Defaults to the current directory', default = '.')
parser.add_argument('-r', '--valid_ratio', help = 'ratio of training images to be used for validation (default 15%%)', default=0.15, type=float)
parser.add_argument('-c', '--category', help='the name of the tod object, defaults to heart_0',default = 'heart_0')
#parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
metafolder = args.meta
metafile = os.path.join(metafolder,'inference_meta.yaml')
diameter_file = os.path.join(metafolder,'diameter.txt')
model_file = os.path.join(metafolder,args.category+'.obj')
ply_file = os.path.join(metafolder,args.category+'.ply')
K, corner_3d, center_3d, fps_3d = read_inference_meta(metafile)
#fps_3d = np.asarray([[4.169e-3,2.1632e-2,-9.52e-4],[-7.708e-3,-3.2392e-2,-7.17e-4],[0.034255 ,-0.000747, 0.000808],[-0.031651, 0.012909, -0.000851], [0.02077, -0.01981, 0.], [-0.02077, -0.01981, 0.]])
center_3d = center_3d.T
K = np.array(K)

obj = utils.MeshObj()
obj.read_obj(model_file,1000)

visualize = args.visualize
test_texture = args.test_texture
valid_ratio = args.valid_ratio

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_test = os.path.join(output_dir, 'test.json')
output_train = os.path.join(output_dir, 'train.json')
output_valid = os.path.join(output_dir, 'test_val.json')


test_json = {}
test_images = []
test_annotations = []
test_categories = []
test_categories.append({
    "supercategory": "none",
    "id": 1,
    "name": args.category
})

train_json = {}
train_images = []
train_annotations = []
train_categories = []
train_categories.append({
    "supercategory": "none",
    "id": 1,
    "name": args.category
})

valid_json = {}

random.seed(42)

train_id = 0
test_id = 0

for texture in range(0,10):
    for pose in poses_for_texture(texture, args.category):
        folder_name = 'texture_'+str(texture)+'_pose_'+str(pose)
        print('Processing',folder_name)
        for image_i in range(0,80):
            if texture == test_texture:
                process_image(folder_name, image_i, test_id, test_images, test_annotations, args.category)
                test_id = test_id+1
            else:
                process_image(folder_name, image_i, train_id, train_images, train_annotations, args.category)
                train_id=train_id+1

scramble_list = list(zip(train_images,train_annotations))
random.shuffle(scramble_list)
train_images, train_annotations = zip(*scramble_list)

split_index = int(len(train_images)*valid_ratio)

valid_images = train_images[:split_index]
valid_annotations = train_annotations[:split_index]

train_images = train_images[split_index:]
train_annotations = train_annotations[split_index:]

test_json["images"] = test_images
test_json["annotations"] = test_annotations
test_json["categories"] = test_categories

train_json["images"] = train_images
train_json["annotations"] = train_annotations
train_json["categories"] = train_categories

valid_json["images"] = valid_images
valid_json["annotations"] = valid_annotations
valid_json["categories"] = train_categories

with open(output_test, 'w') as json_file:
    json.dump(test_json, json_file, 
                        indent=4,  
                        separators=(',',': '))

with open(output_train, 'w') as json_file:
    json.dump(train_json, json_file, 
                        indent=4,  
                        separators=(',',': '))

with open(output_valid, 'w') as json_file:
    json.dump(valid_json, json_file, 
                        indent=4,  
                        separators=(',',': '))

shutil.copy(metafile, output_dir)
shutil.copy(diameter_file, output_dir)
shutil.copy(ply_file, os.path.join(output_dir,'model.ply'))

