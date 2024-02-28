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
parser.add_argument('-d','--data_dir', type=str, help='data directory')
parser.add_argument('-o','--object', type=str, help='object name')
parser.add_argument('-v','--val_percent', type=float, help='percentage of input images for the validation set')
parser.add_argument('-t','--test_percent', type=float, help='percentage of input images for the test set')

if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = args.data_dir
    object_dirs = [os.path.join(data_dir,x) for x in os.listdir(args.data_dir) if x.startswith(args.object) and x.endswith("_bg")]
    bg_dirs = [os.path.join(data_dir,"_".join(x.split("_")[-2:])) for x in object_dirs]
    output_dir = os.path.join(data_dir, args.object)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print(object_dirs)
    print(bg_dirs)

    obj_class = args.object
    obj_name = os.path.join(data_dir, "models/"+obj_class+".ply")
    if not os.path.exists(obj_name):
        raise  ValueError("Object file not found: {}".format(obj_name))
    shutil.copyfile(obj_name, os.path.join(output_dir, "model.ply"))
    object_cloud = pvnet_data_utils.get_ply_model(obj_name)
    object_cloud = object_cloud/1000
    print(object_cloud.shape)

    centroid = np.mean(object_cloud, axis=0)
    diffx = np.max(object_cloud[:,0]) - np.min(object_cloud[:,0])
    diffy = np.max(object_cloud[:,1]) - np.min(object_cloud[:,1])
    diffz = np.max(object_cloud[:,2]) - np.min(object_cloud[:,2])
    diameter = max(diffx, diffy, diffz)
    print(diameter)
    with open(os.path.join(output_dir, "diameter.txt"), "w") as f:
        f.write(str(diameter))
    
    fps_points = fps_utils.farthest_point_sampling(object_cloud, 8, True)
    #fps_points = np.concatenate([fps_points, centroid[None,:]], axis=0)
    print(fps_points.shape)

    camera_mat = cv2.FileStorage(os.path.join(data_dir,"calib_rectified.yml"), cv2.FILE_STORAGE_READ)
    camera_mat = camera_mat.getNode("K").mat()

    print(camera_mat)

    
    left_imgs = []
    tot_instances = 0
    for bg_dir in bg_dirs:
        left_imgs.append(
            [x for x  in os.listdir(os.path.join(bg_dir, "left")) if x.endswith(".png")]
        )
        tot_instances += len(left_imgs[-1])

    print("tot instances", tot_instances)

    idxs = list(range(tot_instances))
    random.shuffle(idxs)
    val_idxs = idxs[:int(len(idxs)*args.val_percent)]
    test_idxs = idxs[int(len(idxs)*args.val_percent):int(len(idxs)*(args.val_percent+args.test_percent))]
    train_idxs = idxs[int(len(idxs)*(args.val_percent+args.test_percent)):]

    train_json = {"images": [], "annotations": []}
    test_json = {"images": [], "annotations": []}
    val_json = {"images": [], "annotations": []}

    for dir_i, (obj_dir, bg_dir) in enumerate(zip(object_dirs, bg_dirs)):
        for img_i in range(len(left_imgs[dir_i])):
            if dir_i > 0:
                shuffle_index = dir_i * len(left_imgs[dir_i -1]) + img_i
            else:
                shuffle_index = img_i
            img_name = "img_{:06d}.png".format(img_i)
            mask_name = "mask_{:06d}.png".format(img_i)
            pose_name = "pose_{:06d}.yml".format(img_i)
            left_img = os.path.abspath(os.path.join(bg_dir, "left", img_name))
            right_img = os.path.abspath(os.path.join(bg_dir, "right", img_name))
            left_mask = os.path.abspath(os.path.join(obj_dir, "left_mask", mask_name))
            right_mask = os.path.abspath(os.path.join(obj_dir, "right_mask", mask_name))
            gt_pose = os.path.abspath(os.path.join(obj_dir, "gt_poses", pose_name))

        #print(gt_pose)



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
            img_obj["id"] = shuffle_index

            ann_obj = {}
            ann_obj["id"] = shuffle_index
            ann_obj["image_id"] = shuffle_index
            ann_obj["category_id"] = 1
            ann_obj["mask_path_L"] = left_mask
            ann_obj["mask_path_R"] = right_mask
            ann_obj["pose_L"] = gt_pose.tolist()
            ann_obj["pose_R"] = gt_pose.tolist()
            ann_obj["baseline"] = 0.117341
            ann_obj["data_root"] = data_dir
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

            if shuffle_index in val_idxs:
                val_json["images"].append(img_obj)
                val_json["annotations"].append(ann_obj)
            elif shuffle_index in test_idxs:
                test_json["images"].append(img_obj)
                test_json["annotations"].append(ann_obj)
            else:
                train_json["images"].append(img_obj)
                train_json["annotations"].append(ann_obj)

            visualize = False
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
    
    print("train",len(train_json["images"]))
    print("test",len(test_json["images"]))
    print("val",len(val_json["images"]))
    
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_json, f)
    
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test_json, f)
    
    with open(os.path.join(output_dir, "test_val.json"), "w") as f:
        json.dump(val_json, f)



        
