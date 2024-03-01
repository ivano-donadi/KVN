import numpy as np
import cv2
import lib.evaluators.custom.iterative_pnp_stereo as ip
import lib.evaluators.custom.iterative_pnp_stereo_variance as ipv
import lib.evaluators.custom.iterative_pnp_stereo_prob as ipp
#from lib.csrc.uncertainty_pnp import un_pnp_utils

def initial_guess_from_disparity(K, kpt_3d, kpt_2d_L, kpt_2d_R, baseline):
    # Z = b* f / (x_l - x_r)
    # K 
    #inf_dist = False
    #est_kpt_3d = np.zeros(kpt_3d.shape)
    #f = K[0][0]
    #for i, (pl, pr) in enumerate(zip(kpt_2d_L, kpt_2d_R)):
    #    y_diff = pl[1] - pr[1]
    #    d = pl[0] - pr[0]
    #    if d == 0 or y_diff > 10:
    #        inf_dist = True
    #        break
    #    Z = (baseline * f)/d
    #    X = (baseline * pl[0])/d
    #    Y = (baseline * pl[1])/d
    #    est_kpt_3d[i][0] = X
    #    est_kpt_3d[i][1] = Y
    #    est_kpt_3d[i][2] = Z
    ## if the distance is infinite (or the y offset is too high) we have bad initialization so we start from a null transformation
    #if inf_dist:
    t = np.asarray([0.,0.,1])
    return (np.eye(3), t)

    #mean1 = np.reshape(np.mean(kpt_3d,0),(3,1))
    #mean2 = np.reshape(np.mean(est_kpt_3d,0),(3,1))
    #H = (kpt_3d.T - mean1).dot((est_kpt_3d.T-mean2).T)
    #U,_,V = np.linalg.svd(H)
    #R = V.dot(U.T)
    #t = mean2 - R.dot(mean1)
    #return (R,t)

def iterative_pnp(K, baseline, img_width, img_height, kpt_2d_L, kpt_2d_R, var_L, var_R, kpt_3d, initial_guess = None, use_variance=True):
    #initial_guess = un_pnp(kpt_2d_L, kpt_3d, var_L, K)
    if initial_guess is not None:
        initial_R, initial_t = (initial_guess[:,:3], initial_guess[:,3])
    else:
        initial_R, initial_t = initial_guess_from_disparity(K, kpt_3d, kpt_2d_L, kpt_2d_R, baseline) 
    
   
    #print("Initial R guess", initial_R)
    #print("Initial t guess", initial_t)
    if use_variance:
        PNPSolver = ipv.PNPSolverVariance()
    else:
        PNPSolver = ip.PNPSolver()
    PNPSolver.setCamModel(K, img_width, img_height, baseline)
    PNPSolver.setInitialTransformation(initial_R.copy(), initial_t.copy())
    
    if use_variance:
        r_mat, transl = PNPSolver.compute(kpt_3d, kpt_2d_L, kpt_2d_R, var_L, var_R)
    else:
        r_mat, transl = PNPSolver.compute(kpt_3d, kpt_2d_L, kpt_2d_R)
    return r_mat, transl

def iterative_pnp_prob(K, baseline, img_width, img_height, kpt_2d_L, kpt_2d_R, prob_L, prob_R, kpt_3d):
    initial_R, initial_t = initial_guess_from_disparity(K, kpt_3d, kpt_2d_L, kpt_2d_R, baseline) 
    PNPSolver = ipp.PNPSolverProb()
    PNPSolver.setCamModel(K, img_width, img_height, baseline)
    PNPSolver.setInitialTransformation(initial_R.copy(), initial_t.copy())
    r_mat, transl = PNPSolver.compute(kpt_3d, kpt_2d_L, kpt_2d_R, prob_L, prob_R)
    return r_mat, transl

def pnp(points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_ITERATIVE):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method == cv2.SOLVEPNP_EPNP:
        points_3d = np.expand_dims(points_3d, 0)
        points_2d = np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    _, R_exp, t = cv2.solvePnP(points_3d,
                               points_2d,
                               camera_matrix,
                               dist_coeffs,
                               flags=method)
    # , None, None, False, cv2.SOLVEPNP_UPNP)

    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t], axis=-1)

#def un_pnp(kpt_3d, kpt_2d, var, K):
#    cov_invs = []
#    for vi in range(var.shape[0]):
#        if var[vi, 0,0] < 1e-6 or np.sum(np.isnan(var)[vi])>0:
#            cov_invs.append(np.zeros([2,2]).astype(np.float32))
#        else:
#            cov_inv = np.linalg.inv(scipy.linalg.sqrtm(var[vi]))
#            cov_invs.append(cov_inv)
#    cov_invs = np.asarray(cov_invs)
#    weights = cov_invs.reshape([-1,4])
#    weights = weights[:,(0,1,3)]
#    pose_pred = un_pnp_utils.uncertainty_pnp(kpt_2d, weights, kpt_3d,k)
#    return pose_pred

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def cm_degree_5(pose_pred, pose_targets):
    translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
    rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
    return translation_distance, angular_distance
