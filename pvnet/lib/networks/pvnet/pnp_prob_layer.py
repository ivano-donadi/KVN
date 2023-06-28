from operator import gt
from turtle import forward
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Function, Variable, grad
from lib.utils.pvnet.pvnet_pose_utils import iterative_pnp_prob, project, iterative_pnp, pnp
import numpy as np
import cv2

def rotate_rodriguez(kp3d, rvec):
    '''
    kp3d: [b,vn, 3]
    rvec: [b,3]
    '''
    if len(kp3d.shape) == 2:
        kp3d = kp3d.unsqueeze(0)
    rvec = torch.atleast_2d(rvec)
    vn = kp3d.shape[1]
    theta = torch.norm(rvec)
    k = (rvec/theta).unsqueeze(2) # b,3,1
    k_T = k.permute(0,2,1) # b,1,3
    ctheta = torch.cos(theta)
    stheta = torch.sin(theta)
    rotated = kp3d * ctheta + torch.cross(k_T.expand(-1,vn,-1), kp3d, dim=2) * stheta + (k * torch.matmul(k_T, kp3d.permute(0,2,1))).permute(0,2,1) * (1-ctheta)
    return rotated

def initial_rt_guess(kpl, pl, kp3d,K):
    winners = []

    for k in range(kpl.shape[0]):
        winner = torch.multinomial(pl[k],1)
        winner = kpl[k,winner]
        winners.append(winner)
    winners = torch.stack(winners).squeeze(1)
    dist_coeffs = np.zeros((4,1))
    succ, rvec, tvec = cv2.solvePnP(kp3d.detach().cpu().numpy(), winners.detach().cpu().numpy(), K.detach().cpu().numpy(), dist_coeffs) 

    return succ, torch.from_numpy(rvec).squeeze(1), torch.from_numpy(tvec).squeeze(1)

def reproj_for_img(rvec, t, kp, prob, kp_3d, K, baseline):
    cur_kp_3d_proj = rotate_rodriguez(kp_3d,rvec)[0] + torch.t(t)
    cur_kp_3d_proj[:,2] = cur_kp_3d_proj[:,2] - baseline # 0 for left image
    cur_kp_3d_proj = torch.matmul(cur_kp_3d_proj, torch.t(K))
    cur_kp_3d_proj = (cur_kp_3d_proj[:, :2] / cur_kp_3d_proj[:, 2:]).unsqueeze(1)
    diff = (kp - cur_kp_3d_proj).pow(2).sum(-1).sqrt() # vn, rn
    wdiff = diff * prob # vn, rn
    wdiff = wdiff.sum(-1) # vn
    return wdiff.mean()


def objective(rvec, t,kp_L, kp_R, prob_L, prob_R, kp_3d, K, baseline):
    error_L = reproj_for_img(rvec, t, kp_L, prob_L, kp_3d, K, 0.)
    error_R = reproj_for_img(rvec, t, kp_R, prob_R, kp_3d, K, baseline)
    return (error_L + error_R)/2.

def Dy(x, DYf, DYYf):
    """
    Dy(x) = -(D_YY^2 f(x, y))^-1 D_XY^2 f(x, y)
    Lemma 4.3 from
    Stephen Gould, Richard Hartley, and Dylan Campbell, 2019
    , arXiv:1909.04866
    Arguments:
        f: (b, ) Torch tensor, with gradients
            batch of objective functions evaluated at (x, y)
        x: (b, n) Torch tensor, with gradients
            batch of input vectors
        y: (b, m) Torch tensor, with gradients
            batch of minima of f
    Return Values:
        Dy(x): (b, m, n) Torch tensor,
            batch of gradients of y with respect to x
    """
    with torch.enable_grad():
        if len(x.shape) > 2:
            DXYf = torch.empty_like(DYf.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, x.shape[0],x.shape[1],x.shape[2])) # R: 9xnkpxnhpx2, t:3xnkpxnhpx2
        else:
            DXYf = torch.empty_like(DYf.unsqueeze(1).unsqueeze(2).expand(-1, x.shape[0],x.shape[1])) # R: 9xnkpxnhp, t:3xnkpxnhp
        
        grad_outputs = torch.ones_like(DYf[0])
        for i in range(DYf.shape[-1]):
            DXYf[i,:,:] = grad(DYf[i], x, grad_outputs = grad_outputs,create_graph=True)[0]
        DXYf = DXYf.flatten(1)

    # m = 9
    # n = nkp x nhp x 2
    DYYf = DYYf.detach().unsqueeze(0)
    DXYf = DXYf.detach().unsqueeze(0)    
    DYYf = 0.5 * (DYYf + DYYf.transpose(1, 2)) # In case of floating point errors

    # Try a batchwise solve, otherwise revert to looping
    # Avoids cuda runtime error (9): invalid configuration argument
    try:
        U = torch.cholesky(DYYf, upper=True)
        Dy_at_x = torch.cholesky_solve(-1.0 * DXYf, U, upper=True) # bxmxn
    except:
        Dy_at_x = torch.empty_like(DXYf)
        for i in range(DYYf.size(0)): # For some reason, doing this in a loop doesn't crash
            try:
                U = torch.cholesky(DYYf[i, ...], upper=True)
                Dy_at_x[i, ...] = torch.cholesky_solve(-1.0 * DXYf[i, ...], U, upper=True)
            except:
                Dy_at_x[i, ...], _ = torch.solve(-1.0 * DXYf[i, ...], DYYf[i, ...])

    # Set NaNs to 0:
    if torch.isnan(Dy_at_x).any():
        Dy_at_x[torch.isnan(Dy_at_x)] = 0.0 # In-place
    # Clip gradient norms:
    max_norm = 100.0
    Dy_norm = Dy_at_x.norm(dim=-2, keepdim=True) # bxmxn
    if (Dy_norm > max_norm).any():
        clip_coef = (max_norm / (Dy_norm + 1e-6)).clamp_max_(1.0)
        Dy_at_x = clip_coef * Dy_at_x

    return Dy_at_x

class PnPFunction(Function):

    @staticmethod
    def forward(ctx, kp_L, offset_L, kp_R, offset_R, prob_L, prob_R, kp_3D, K, baseline, img_shape):
        K_cpu = K#.cpu().numpy()
        prob_L_cpu = prob_L.detach().permute(0,2,1)#.cpu().numpy()
        prob_R_cpu = prob_R.detach().permute(0,2,1)#.cpu().numpy()
        kp_L_cpu = kp_L.detach().permute(0,2,1,3)
        kp_R_cpu = kp_R.detach().permute(0,2,1,3)
        kp_L_cpu = (kp_L_cpu + offset_L[:, None, None, :])#.cpu().numpy()
        kp_R_cpu = (kp_R_cpu + offset_R[:, None, None, :])#.cpu().numpy()
        kp_3D_cpu = kp_3D.detach()#.cpu().numpy()

        batch_size = kp_L_cpu.shape[0]
        all_r, all_t = [],[]

        for b in range(batch_size):
            kl = kp_L_cpu[b]
            kr = kp_R_cpu[b]
            pl = prob_L_cpu[b]
            pr = prob_R_cpu[b]
            kp3d = kp_3D_cpu[b]
            succ, rvec, t= initial_rt_guess(kl,pl,kp3d,K_cpu)
            if not succ:
                rvec = torch.zeros(3)
                t = torch.zeros_like(rvec)
                t[0] = 1.
            rvec = rvec.float().cuda()
            t = t.float().cuda()

            if torch.is_grad_enabled():
                rvec.requires_grad_()
                t.requires_grad_()
            opt = torch.optim.LBFGS([rvec, t],
                                    lr=1.0, # Default: 1
                                    max_iter=100, # Default: 100
                                    max_eval=None,
                                    tolerance_grad=1e-05, #1e-40
                                    tolerance_change= 1e-09, #1e-40
                                    history_size=100,
                                    line_search_fn="strong_wolfe",
                                    )
            def closure():
                if torch.is_grad_enabled():
                    opt.zero_grad()
                error = objective(rvec, t,kl, kr, pl, pr, kp3d, K_cpu, baseline)
                if error.requires_grad:
                    error.backward()
                return error
            opt.step(closure)
            rvec = rvec.detach()
            t = t.detach()
            #rmat, t = iterative_pnp_prob(K_cpu, baseline, img_shape[1], img_shape[0], kl, kr, pl, pr, kp3d)
            all_r.append(rvec)
            all_t.append(t)

        rvecs = torch.stack(all_r)
        tvecs = torch.stack(all_t)
        ctx.save_for_backward(kp_L, kp_R, prob_L, prob_R, rvecs, tvecs)
        ctx.K = K
        ctx.baseline = baseline
        ctx.kp_3d = kp_3D
        ctx.offset_L = offset_L
        ctx.offset_R = offset_R
        return rvecs, tvecs

    @staticmethod
    def backward(ctx, grad_R, grad_t):
        grad_kp_L = grad_off_L = grad_kp_R = grad_off_R = grad_prob_L = grad_prob_R = grad_kp_3d = grad_K = grad_baseline = grad_img_shape = None
        kp_L, kp_R, prob_L, prob_R, R, t = ctx.saved_tensors # [b, rn, vn, 2], [b, rn, vn, 2], [b, rn, vn], [b, rn, vn]
        K = ctx.K
        baseline = ctx.baseline
        kp_3d = ctx.kp_3d
        offset_L = ctx.offset_L
        offset_R = ctx.offset_R
        kp_L_det = kp_L.detach().permute(0,2,1,3) + offset_L[:, None, None, :]
        kp_R_det = kp_R.detach().permute(0,2,1,3) + offset_R[:, None, None, :]
        prob_L_det = prob_L.detach().permute(0,2,1)
        prob_R_det = prob_R.detach().permute(0,2,1)

        batch_size, nkp, nhp, _ = kp_L_det.shape

        grad_kp_L = []
        grad_kp_R = []
        grad_prob_L = []
        grad_prob_R = []

        bl = torch.zeros_like(t[0])
        bl[0] = -baseline
        bl = bl.unsqueeze(1)

        for b in range(batch_size):
            with torch.enable_grad():
                curr_R,curr_t = R[b].detach().requires_grad_(), t[b].unsqueeze(1).detach().requires_grad_()
                kp_L_b = kp_L_det[b].detach().requires_grad_()
                kp_R_b = kp_R_det[b].detach().requires_grad_()
                p_L_b = prob_L_det[b].detach().requires_grad_()
                p_R_b = prob_R_det[b].detach().requires_grad_()
                f_at_kp_prob = objective(curr_R, curr_t,kp_L_b, kp_R_b, p_L_b, p_R_b, kp_3d[b], K, baseline) # [1]
                grad_outputs = torch.ones_like(f_at_kp_prob)
                DRf = grad(f_at_kp_prob, curr_R, grad_outputs=grad_outputs, create_graph=True)[0].flatten() # R:9, t:3
                Dtf = grad(f_at_kp_prob, curr_t, grad_outputs=grad_outputs, create_graph=True)[0].flatten()
                DRRf = torch.empty_like(DRf.unsqueeze(1).expand(-1, DRf.shape[-1])) # R: 9x9, t:3x3
                Dttf = torch.empty_like(Dtf.unsqueeze(1).expand(-1, Dtf.shape[-1])) # R: 9x9, t:3x3
                grad_outputs_R = torch.ones_like(DRf[0])
                grad_outputs_t = torch.ones_like(Dtf[0])
                for i in range(DRf.shape[-1]):
                    DRRf[i,:] = grad(DRf[i], curr_R, grad_outputs = grad_outputs_R,create_graph=True)[0].flatten()
                for i in range(Dtf.shape[-1]):
                    Dttf[i,:] = grad(Dtf[i], curr_t, grad_outputs = grad_outputs_t,create_graph=True)[0].flatten()
            
            Dkp_L_DR = Dy(kp_L_b, DRf, DRRf)[0] 
            Dkp_R_DR = Dy(kp_R_b, DRf, DRRf)[0]
            Dkp_L_Dt = Dy(kp_L_b, Dtf, Dttf)[0]
            Dkp_R_Dt = Dy(kp_R_b, Dtf, Dttf)[0]
            Dpr_L_DR = Dy(p_L_b, DRf, DRRf)[0]
            Dpr_R_DR = Dy(p_R_b, DRf, DRRf)[0]
            Dpr_L_Dt = Dy(p_L_b, Dtf, Dttf)[0]
            Dpr_R_Dt = Dy(p_R_b, Dtf, Dttf)[0]

            gR = grad_R[b].flatten()
            gt = grad_t[b].flatten()

            grad_kp_L.append(torch.matmul(gR, Dkp_L_DR) + torch.matmul(gt,Dkp_L_Dt))
            grad_kp_R.append(torch.matmul(gR, Dkp_R_DR) + torch.matmul(gt,Dkp_R_Dt))
            grad_prob_L.append(torch.matmul(gR, Dpr_L_DR) + torch.matmul(gt,Dpr_L_Dt))
            grad_prob_R.append(torch.matmul(gR, Dpr_R_DR) + torch.matmul(gt,Dpr_R_Dt))

        

        grad_kp_L = torch.atleast_2d(torch.cat(grad_kp_L,0)).reshape(batch_size, nkp, nhp, 2).permute(0,2,1,3)
        grad_kp_R = torch.atleast_2d(torch.cat(grad_kp_R,0)).reshape(batch_size, nkp, nhp, 2).permute(0,2,1,3)
        grad_prob_L = torch.atleast_2d(torch.cat(grad_prob_L,0)).reshape(batch_size, nkp, nhp).permute(0,2,1)
        grad_prob_R = torch.atleast_2d(torch.cat(grad_prob_R,0)).reshape(batch_size, nkp, nhp).permute(0,2,1)

        return grad_kp_L, grad_off_L, grad_kp_R, grad_off_R, grad_prob_L, grad_prob_R, grad_kp_3d, grad_K, grad_baseline, grad_img_shape

def pnp_loss(rvec,t, pose_gt, kp_3d, kpl, pl, kpr, pr, K, baseline):
    '''
    R: [b,3,3]
    t: [b,3]
    pose_gt: [b,3,4]
    kp_3d: [b,vn,3]
    '''
    R_gt = pose_gt[:,:,:3]
    t_gt = pose_gt[:,:,3]

    transformed = rotate_rodriguez(kp_3d, rvec) + t.unsqueeze(1) # [b, vn, 3]
    kp_3d_gt = torch.bmm(kp_3d, R_gt.permute(0,2,1)) + t_gt.unsqueeze(1) # [b, vn, 3]
    
    diff = (transformed - kp_3d_gt).pow(2).sum(2).sqrt().mean()

    # regularization term
    reg = torch.zeros(1,dtype=torch.float32, device = rvec.device)
    for b in range(rvec.shape[0]):
        reg += objective(rvec[b].detach(), t[b].detach(), kpl[b],kpr[b], pl[b], pr[b], kp_3d[b], K, baseline)
    reg = reg/rvec.shape[0]
    reg = reg[0]

    return diff #+ 1e-2*reg
