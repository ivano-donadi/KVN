import argparse
import os
import numpy as np
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from lib.csrc.fps import fps_utils
from lib.utils.pvnet import pvnet_data_utils, pvnet_pose_utils
import cv2
import yaml
import matplotlib.pyplot as plt
import random
import json
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', type=str, help='input json file')
parser.add_argument('-d','--base_dir', type=str, help='folder where to find the ttd dataset')
parser.add_argument('-n', '--name', type=str, help='name of the object of interest')
parser.add_argument('-o','--output', type=str, help='output folder where to store the pvnet annotations json')

if __name__ == "__main__":
    args = parser.parse_args()

    #### loading input params
    input_json = json.load(open(args.input))
    obj_class = args.name
    base_dir = args.base_dir
    mesh_file = os.path.join(base_dir, "ttd", "models",obj_class+".ply")
    output_dir = os.path.join(args.output, obj_class)
    camera_mat= os.path.join(base_dir, "ttd", "cameras_params.yml") 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok =True)

    print('Input file:', args.input)
    print('Object class:',obj_class)
    print('Mesh file:', mesh_file)
    print('Camera file:', camera_mat)
    print('Output dir:', output_dir)

    #### extracting mesh information
    if not os.path.exists(mesh_file):
        raise  ValueError("Object file not found: {}".format(mesh_file))
    shutil.copyfile(mesh_file, os.path.join(output_dir, "model.ply"))
    object_cloud = pvnet_data_utils.get_ply_model(mesh_file)
    object_cloud = object_cloud/1000
    print('Object point cloud shape:', object_cloud.shape)

    centroid = np.mean(object_cloud, axis=0)
    diffx = np.max(object_cloud[:,0]) - np.min(object_cloud[:,0])
    diffy = np.max(object_cloud[:,1]) - np.min(object_cloud[:,1])
    diffz = np.max(object_cloud[:,2]) - np.min(object_cloud[:,2])
    diameter = max(diffx, diffy, diffz)
    print('Object diameter:',diameter)
    with open(os.path.join(output_dir, "diameter.txt"), "w") as f:
        f.write(str(diameter))
    

    #### generating keypoints
    fps_points = fps_utils.farthest_point_sampling(object_cloud, 8, True)
    print('Furthest point sampling keypoints shape:',fps_points.shape)

    #### loading camera parameters
    camera_mat = cv2.FileStorage(camera_mat, cv2.FILE_STORAGE_READ)
    camera_mat = camera_mat.getNode("K").mat()
    print('K  matrix:',camera_mat)

    #### filtering occluded instances
    print("total instances:", len(input_json))
    print('of which occluded:',len([x for x in input_json if x['ground_truth'][obj_class]['poses']['left_camera'] is None]))
    viable_instances = [x for x in input_json if x['ground_truth'][obj_class]['poses']['left_camera'] is not None]

    #### writing each instance in pvnet's format
    output_json = {"images": [], "annotations": []}
    for index,instance in enumerate(viable_instances):
        left_img = os.path.join(base_dir, instance['stereo_images']['left_camera'])
        right_img =  os.path.join(base_dir, instance['stereo_images']['right_camera'])
        left_mask = os.path.join(base_dir, instance['ground_truth'][obj_class]['masks']['left_camera'])
        right_mask = os.path.join(base_dir, instance['ground_truth'][obj_class]['masks']['right_camera'])
        gt_pose = os.path.join(base_dir, instance['ground_truth'][obj_class]['poses']['left_camera'])

        if not os.path.exists(left_mask) or not os.path.exists(right_mask) or not os.path.exists(left_img) or not os.path.exists(right_img) or not os.path.exists(gt_pose):
            print("Missing image: {}".format(left_img))
            continue

        gt_pose = yaml.safe_load(open(gt_pose))
        gt_rotation = np.array(gt_pose["rotation"]["data"]).reshape(3,3)
        gt_translation = np.array(gt_pose["translation"]["data"]).reshape(3,1)
        gt_pose = np.concatenate([gt_rotation, gt_translation], axis=1)
    

        img_obj = {}
        img_obj["file_name_L"] = left_img
        img_obj["file_name_R"] = right_img
        img_obj["height"] = 768
        img_obj["width"] = 1024
        img_obj["id"] = index

        ann_obj = {}
        ann_obj["id"] = index
        ann_obj["image_id"] = index
        ann_obj["category_id"] = 1
        ann_obj["mask_path_L"] = left_mask
        ann_obj["mask_path_R"] = right_mask
        ann_obj["pose_L"] = gt_pose.tolist()
        ann_obj["pose_R"] = gt_pose.tolist()
        ann_obj["baseline"] = 0.117341
        ann_obj["data_root"] = 'unset'
        ann_obj["type"] = "real"
        ann_obj["cls"] = obj_class
        ann_obj["K"] = camera_mat.tolist()
        ann_obj["fps_3d_L"] = fps_points.tolist()
        ann_obj["center_3d_L"] = centroid.tolist()
        ann_obj["fps_3d_R"] = fps_points.tolist()
        ann_obj["center_3d_R"] = centroid.tolist()
        ann_obj["fps_2d_L"] = pvnet_pose_utils.project(fps_points, camera_mat, gt_pose).tolist()
        ann_obj["center_2d_L"] = pvnet_pose_utils.project(centroid, camera_mat, gt_pose).tolist()
        gt_pose_r = gt_pose.copy()
        gt_pose_r[0,3] -= 0.117341
        ann_obj["fps_2d_R"] = pvnet_pose_utils.project(fps_points, camera_mat, gt_pose_r).tolist()
        ann_obj["center_2d_R"] = pvnet_pose_utils.project(centroid, camera_mat, gt_pose_r).tolist()

        
        output_json["images"].append(img_obj)
        output_json["annotations"].append(ann_obj)
        visualize = True
        if visualize:
            print(gt_pose)
            gt_pose[0,3] -= 0.117341
            fps_2d = pvnet_pose_utils.project(fps_points, camera_mat, gt_pose)
            print(fps_2d.shape)
            
            #quit()
            plt.imshow(cv2.imread(right_img))
            plt.imshow(cv2.imread(right_mask), alpha=0.5)
            plt.plot(fps_2d[:,0], fps_2d[:,1], 'ro')
            plt.show()

    #### saving pvnet annotations
    with open(os.path.join(output_dir, os.path.basename(args.input)), "w") as f:
        json.dump(output_json, f)



        
