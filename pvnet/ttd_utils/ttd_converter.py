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

def orthogonal_x_rotation(a, b, r, rcol):

  A = r[0,rcol]*a[2,0] + r[1,rcol]*a[2,1]
  B = -r[0,rcol]*b[2,0] - r[1,rcol]*b[2,1] - r[2,rcol]*b[2,2]
  C = r[0,rcol]*b[2,0] + r[1,rcol]*b[2,1] + r[2,rcol] + r[2,rcol]*b[2,2]

  D = r[0,rcol] + b[0,0]*r[0,rcol] + b[0,1]*r[1,rcol] + b[0,2]*r[2,rcol]
  E = a[0,1]*r[1,rcol] + a[0,2]*r[2,rcol]
  F = -r[0,rcol]*b[0,0] - b[0,1]*r[1,rcol] - b[0,2]*r[2,rcol]

  G = r[0,rcol]*b[1,0] + b[1,1]*r[1,rcol] + r[1,0] + b[1,2]*r[2,rcol]
  H = a[1,0]*r[0,rcol] + a[1,2]*r[2,rcol]
  I = -r[0,rcol]*b[1,0] - r[1,rcol]*b[1,1] - r[2,rcol]*b[1,2]

  if np.abs(B) < 1e-10:
    print('0 B')
    if np.abs(A) < 1e-10:
      print('0 B,A')
      theta = 0
      theta_2 = 0  
    else:
      s_theta = -C/A
      theta = np.arcsin(s_theta)
      theta_2 = theta
  else:
    J = D - F*C/B
    K = E - F*A/B
    L = G - I*C/B
    M = H - I*A/B

    alpha = K*K + M*M
    beta = 2*J*K + 2*L*M
    gamma = J*J + L*L - 1

    if np.abs(alpha) < 1e-10:
      print('0 alpha')
      if np.abs(beta) < 1e-10:
          print('0 alpha, beta')
          theta = 0
          theta_2 = 0
      else:
          s_theta = -gamma/beta
          c_theta = - A * s_theta / B - C/B
          theta = np.arctan2(s_theta, c_theta)
          theta_2 = theta
    else:
      if np.abs(beta) < 1e-10:
        s_theta = np.sqrt(-gamma/alpha)
        s_theta_2 = - s_theta
      else:
        s_theta = (beta + np.sqrt(beta*beta-4*alpha*gamma))/(2*alpha)
        s_theta_2 = (beta - np.sqrt(beta*beta-4*alpha*gamma))/(2*alpha)
      c_theta = - A * s_theta / B - C/B
      c_theta_2 = - A * s_theta_2 / B - C/B 
      theta = np.arctan2(s_theta, c_theta)
      theta_2 = np.arctan2(s_theta_2, c_theta_2)
  return theta, theta_2

def rot_mat_from_theta(theta, a_skew, a_2):
  s_theta = np.sin(theta)
  c_theta = np.cos(theta)
  rmat = np.eye(3) + s_theta * a_skew + (1-c_theta) * a_2
  return rmat

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', type=str, help='input json file')
parser.add_argument('-d','--base_dir', type=str, help='folder where to find the ttd dataset')
parser.add_argument('-n', '--name', type=str, help='name of the object of interest')
parser.add_argument('-o','--output', type=str, help='output folder where to store the pvnet annotations json')
parser.add_argument('-s', '--symmetric', action='store_true', dest='symmetric', default=False, help='set the flag  if the object has 360 deg symmetry')

if __name__ == "__main__":
    args = parser.parse_args()

    #### loading input params
    input_json = json.load(open(args.input))
    obj_class = args.name
    base_dir = args.base_dir
    mesh_file = os.path.join(base_dir, "ttd", "models",obj_class+".ply")
    output_dir = os.path.join(args.output, obj_class)
    camera_mat= os.path.join(base_dir, "ttd", "cameras_params.yml") 
    is_symmetric= args.symmetric
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok =True)

    print('Input file:', args.input)
    print('Object class:',obj_class)
    print('Mesh file:', mesh_file)
    print('Camera file:', camera_mat)
    print('Output dir:', output_dir)
    print('Symmetric obj:', is_symmetric)

    #### extracting mesh information
    if not os.path.exists(mesh_file):
        raise  ValueError("Object file not found: {}".format(mesh_file))
    shutil.copyfile(mesh_file, os.path.join(output_dir, "model.ply"))
    object_cloud = pvnet_data_utils.get_ply_model(mesh_file)
    object_cloud = object_cloud/1000
    print('Object point cloud shape:', object_cloud.shape)

    centroid = np.mean(object_cloud, axis=0)
    maxx, minx =  np.max(object_cloud[:,0]),  np.min(object_cloud[:,0])
    maxy, miny =  np.max(object_cloud[:,1]),  np.min(object_cloud[:,1])
    maxz, minz =  np.max(object_cloud[:,2]),  np.min(object_cloud[:,2])
    diffx = maxx - minx
    diffy = maxy - miny
    diffz = maxz - minz
    diameter = max(diffx, diffy, diffz)
    print('Object diameter:',diameter)
    with open(os.path.join(output_dir, "diameter.txt"), "w") as f:
        f.write(str(diameter))
    corners_3d = np.array([
        [minx, miny, minz],
        [minx, miny, maxz],
        [minx, maxy, minz],
        [minx, maxy, maxz],
        [maxx, miny, minz],
        [maxx, miny, maxz],
        [maxx, maxy, minz],
        [maxx, maxy, maxz]
    ])
    

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
        #### standardizing symmetric pose
        if is_symmetric:
            a = gt_pose[:3,1]
            rcol = 0
            a_skew = np.array([[0., -a[2], a[1]], [a[2],0.,-a[0]], [-a[1], a[0], 0.]])
            a_2 = np.dot(a_skew, a_skew)
            theta, theta_2 = orthogonal_x_rotation(a_skew, a_2, gt_pose, rcol)
            rmat = rot_mat_from_theta(theta, a_skew, a_2)
            aligned_rotation = np.dot(rmat, gt_pose[:3,:3])
            alt_condition = aligned_rotation[2,1] > 0
            if alt_condition:
                rmat = rot_mat_from_theta(theta_2, a_skew, a_2)
                aligned_rotation = np.dot(rmat, gt_pose[:3,:3])
            if np.abs(aligned_rotation[2,0]) > 1e-10:
                raise RuntimeError("Symmetric object not correctly aligned")
            gt_pose[:3,:3] = aligned_rotation


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
        ann_obj["corner_3d_L"] =corners_3d.tolist()
        ann_obj["corner_3d_R"] =corners_3d.tolist()
        ann_obj["fps_2d_L"] = pvnet_pose_utils.project(fps_points, camera_mat, gt_pose).tolist()
        ann_obj["center_2d_L"] = pvnet_pose_utils.project(centroid, camera_mat, gt_pose).tolist()
        ann_obj["corner_2d_L"] = pvnet_pose_utils.project(corners_3d, camera_mat, gt_pose).tolist()
        gt_pose_r = gt_pose.copy()
        gt_pose_r[0,3] -= 0.117341
        ann_obj["fps_2d_R"] = pvnet_pose_utils.project(fps_points, camera_mat, gt_pose_r).tolist()
        ann_obj["center_2d_R"] = pvnet_pose_utils.project(centroid, camera_mat, gt_pose_r).tolist()
        ann_obj["corner_2d_R"] = pvnet_pose_utils.project(corners_3d, camera_mat, gt_pose_r).tolist()

        
        output_json["images"].append(img_obj)
        output_json["annotations"].append(ann_obj)
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

    #### saving pvnet annotations
    with open(os.path.join(output_dir, os.path.basename(args.input)), "w") as f:
        json.dump(output_json, f)



        
