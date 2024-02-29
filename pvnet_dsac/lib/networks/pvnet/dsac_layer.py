from operator import gt
from turtle import forward
import torch
import torch.nn as nn
from torch.autograd import Function

thresh = 1e-6
invalid_loss = 1e-20
invalid_score = -1e9
kp_loss_clamp = 1e-1
score_loss_clamp = 1e-1

def suppress_nan_inf(tensor):
    tensor[tensor.isnan()] = 0.
    tensor[tensor.isinf()] = 0.
    return tensor

def suppress_thresh(tensor, comp_tensor, t):
    tensor[comp_tensor.abs() < t] = 0.
    return tensor

def suppress_nan_inf_thresh(tensor, comp_tensor, t):
    tensor = suppress_nan_inf(tensor)
    tensor = suppress_thresh(tensor, comp_tensor, t)
    return tensor

def limit_to_range(tensor, control, min, max):
    tensor[control < min] = 0.
    tensor[control > max] = 0.
    return tensor

def score_function(b,t,distance):
    #return torch.sigmoid(b*(t - distance))
    score = torch.empty_like(distance)
    score[distance <= t] = ((b-1)/t)*distance[distance<=t] + 1
    score[distance > t ] = (b/(t-2))*distance[distance > t] + ((2*b)/(2-t))
    return score

def score_derivative(b,t,dist):
    #return - 1 * (-b*(t-dist)).exp() * torch.sigmoid(b*(t-dist)).pow(2) * b
    grad = torch.empty_like(dist)
    grad[dist <= t] = (b-1)/t
    grad[dist > t] = b/(t-2)
    return grad

def intersection_from_dir(p0,p1,n0,n1):
    cx0 = p0[:,:,0] #[rn,kn]
    cy0 = p0[:,:,1] #[rn,kn]
    cx1 = p1[:,:,0] #[rn,kn]
    cy1 = p1[:,:,1] #[rn,kn]
    nx0 = n0[:,:,1] #[rn,kn]
    ny0 = -n0[:,:,0] #[rn,kn]
    nx1 = n1[:,:,1] #[rn,kn]
    ny1 = -n1[:,:,0] #[rn,kn]

    
    numx = (ny1*(nx0*cx0+ny0*cy0)-ny0*(nx1*cx1+ny1*cy1))
    numy = (nx0*(nx1*cx1+ny1*cy1)-nx1*(ny0*cy0+nx0*cx0))
    denx = (ny1*nx0-ny0*nx1)
    deny = denx
    y=numy/(deny)
    x=numx/(denx)

    #print(denx.max(), deny.max())

    invalidx = x.isnan().logical_or(x.isinf()).logical_or(x <-224).logical_or(x > 2*224)
    invalidy = y.isnan().logical_or(y.isinf()).logical_or(y <-126).logical_or(y > 2*126)
    invalidden = (denx.abs()<thresh).logical_or(deny.abs()<thresh)
    valid = (invalidx.logical_not()).logical_and((invalidy.logical_not())).logical_and(invalidden.logical_not())

    x[valid.logical_not()] = 0
    y[valid.logical_not()] = 0

    #y = suppress_nan_inf_thresh(y,deny,thresh)
    #x = suppress_nan_inf_thresh(x,denx,thresh)

    return x,y,valid

def dh_dv(p0,p1,n0,n1):
    # dl_dh: [rn, kn, 2]
    cx0 = p0[:,:,0] #[rn, kn]
    cy0 = p0[:,:,1] #[rn, kn]
    cx1 = p1[:,:,0] #[rn, kn]
    cy1 = p1[:,:,1] #[rn, kn]
    nx0 = n0[:,:,1] #[rn, kn]
    ny0 = -n0[:,:,0] #[rn, kn]
    nx1 = n1[:,:,1] #[rn, kn]
    ny1 = -n1[:,:,0] #[rn, kn]

    ###numy = (nx1*(nx0*cx0+ny0*cy0)-nx0*(nx1*cx1+ny1*cy1))
    ###numx = (ny1*(nx0*cx0+ny0*cy0)-ny0*(nx1*cx1+ny1*cy1))
    ###deny = (nx1*ny0-nx0*ny1)
    ###denx = (ny1*nx0-ny0*nx1)

    ###dhx_dn0x = (ny1*cx0*denx - numx*ny1)/((denx**2))
    ###dhx_dn0y = ((ny1*cy0-nx1*cx1-ny1*cy1)*denx + numx*nx1)/((denx**2))
    ###dhx_dn1x = ((-ny0*cx1)*denx+ny0*numx)/((denx**2))
    ###dhx_dn1y = ((nx0*cx0+ny0*cy0-ny0*cy1)*denx - numx*nx0)/((denx**2))

    ###dhy_dn0x = ((nx1*cx0-nx1*cx1-ny1*cy1)*deny + numy*ny1)/((deny**2))
    ###dhy_dn0y = (nx1*cy0*deny - numy * nx1)/((deny**2))
    ###dhy_dn1x = ((nx0*cx0+ny0*cy0-nx0*cx1)*deny - numy*ny0)/((deny**2))
    ###dhy_dn1y = (-nx0*cy1*deny + numy*nx0)/((deny**2))

    numx = (ny1*(nx0*cx0+ny0*cy0)-ny0*(nx1*cx1+ny1*cy1))
    numy = (nx0*(nx1*cx1+ny1*cy1)-nx1*(ny0*cy0+nx0*cx0))
    denx = (ny1*nx0-ny0*nx1)
    deny = denx

    dhx_dn0x = (ny1*cx0*denx - numx*ny1)/((denx**2))
    dhx_dn0y = ((ny1*cy0-nx1*cx1-ny1*cy1)*denx + numx*nx1)/((denx**2))
    dhx_dn1x = ((-ny0*cx1)*denx+ny0*numx)/((denx**2))
    dhx_dn1y = ((nx0*cx0+ny0*cy0-ny0*cy1)*denx - numx*nx0)/((denx**2))

    dhy_dn0x = ((nx1*cx1+ny1*cy1-nx1*cx0)*deny - numy*ny1)/((deny**2))
    dhy_dn0y = (-nx1*cy0*deny + numy * nx1)/((deny**2))
    dhy_dn1x = ((nx0*cx1-ny0*cy0-nx0*cx0)*deny + numy*ny0)/((deny**2))
    dhy_dn1y = (nx0*cy1*deny - numy*nx0)/((deny**2))


    dhx_dn0x = suppress_nan_inf(dhx_dn0x)
    dhx_dn1x = suppress_nan_inf(dhx_dn1x)
    dhx_dn0y = suppress_nan_inf(dhx_dn0y)
    dhx_dn1y = suppress_nan_inf(dhx_dn1y)
    dhy_dn0x = suppress_nan_inf(dhy_dn0x)
    dhy_dn1x = suppress_nan_inf(dhy_dn1x)
    dhy_dn0y = suppress_nan_inf(dhy_dn0y)
    dhy_dn1y = suppress_nan_inf(dhy_dn1y)

    # vox = -n0y, v0y = n0x 

    # [ dhx_dvox, dhy_dvox ] [dl_dhx] = [dhx_dv0x * dl_dhx + dhy_dvox * dl_dhy] = [dl_dv0x]
    # [ dhx_dv0y, dhy_dvoy ] [dl_dhy]   [dhx_dvoy * dl_dhx + dhy_dvoy * dl_dhy]   [dl_dv0y]
    dh_dv0 = torch.empty_like(dhx_dn0x.unsqueeze(2).unsqueeze(3).expand(-1,-1,2,2))
    dh_dv1 = torch.empty_like(dh_dv0)
    dh_dv0[:,:,0,0] = -1*dhx_dn0y
    dh_dv0[:,:,0,1] = -1*dhy_dn0y
    dh_dv0[:,:,1,0] = dhx_dn0x
    dh_dv0[:,:,1,1] = dhy_dn0x
    dh_dv1[:,:,0,0] = -1*dhx_dn1y
    dh_dv1[:,:,0,1] = -1*dhy_dn1y
    dh_dv1[:,:,1,0] = dhx_dn1x
    dh_dv1[:,:,1,1] = dhy_dn1x

    return dh_dv0, dh_dv1
    
def dl_dv(p0,p1,n0,n1, dl_dh):

    dh_dv0, dh_dv1 = dh_dv(p0,p1,n0,n1) 
    
    dl_dv0 = torch.einsum('rkij,rkj->rki',dh_dv0, dl_dh) # rn, kn, 2
    dl_dv1 = torch.einsum('rkij,rkj->rki',dh_dv1, dl_dh) # rn, kn, 2

    return (dl_dv0,dl_dv1) 

def ds_dv(p,n,h,a,b,t, dh_dv):
    cx = p[None,:,None,0] # [1, tn, 1]
    cy = p[None,:,None,1] # [1, tn, 1]
    vx = n[None,:,:,0] # [1, tn, kn]
    vy = n[None,:,:,1] # [1, tn, kn]
    hx = h[:, None,:,0] # [rn, 1, kn]
    hy = h[:, None,:,1] # [rn, 1, kn]
    dx=hx-cx # [rn, tn, kn]
    dy=hy-cy # [rn, tn, kn]
    
    ###### N.B: we are ignoring the component of hypothesis loss in the score loss
    #dhx_dvx = dh_dv[:,:,:,0,0] #[rn, tn, kn]
    #dhy_dvx = dh_dv[:,:,:,0,1] #[rn, tn, kn]
    #dhx_dvy = dh_dv[:,:,:,1,0] #[rn, tn, kn]
    #dhy_dvy = dh_dv[:,:,:,1,1] #[rn, tn, kn]

    ###### N.B: we are assuming that v is normalized by the network
    #normv=torch.sqrt(vx*vx+vy*vy).expand(dx.shape[0],-1,-1) # [rn, tn ,kn]
    
    normd=torch.sqrt(dx*dx+dy*dy) # [rn, tn ,kn]
    vd = vx*dx + vy*dy
    similarity = vd/normd
    dist = suppress_nan_inf(1-similarity)
    dsigm_ddist = score_derivative(b,t,dist)

    ds_dvx = dsigm_ddist * -1 * ((dx - vd*vx)/normd)
    ds_dvy = dsigm_ddist * -1 * ((dy - vd*vy)/normd)

    ds_dvx = suppress_nan_inf(ds_dvx)
    ds_dvy = suppress_nan_inf(ds_dvy)
    ds_dvx = ds_dvx.permute(2,0,1).unsqueeze(3) #[kn, rn, tn, 1]
    ds_dvy = ds_dvy.permute(2,0,1).unsqueeze(3) #[kn, rn, tn, 1]
    ######

    ds_dt = None
    ds_db = None

    return torch.cat([ds_dvx, ds_dvy],dim=3), ds_db, ds_dt #[kn, rn, tn, 2], [kn, rn]

def similarity_for_hyp_batched(p,n,h,b,t):
    # tn = number of coordinates, kn = number of keypoints, rn = number of rounds
    cx = p[None,:,None,0] # [1, tn, 1]
    cy = p[None,:,None,1] # [1, tn, 1]
    vx = n[None,:,:,0] # [1, tn, kn]
    vy = n[None,:,:,1] # [1, tn, kn]
    hx = h[:, None,:,0] # [rn, 1, kn]
    hy = h[:, None,:,1] # [rn, 1, kn]
    dx=hx-cx # [rn, tn, kn]
    dy=hy-cy # [rn, tn, kn]
    norm2=torch.sqrt(dx*dx+dy*dy) # [rn, tn ,kn]
    similarity = (dx*vx+dy*vy)/norm2
    similarity[norm2<thresh] = 0
    distance = suppress_nan_inf(1-similarity)
    result = score_function(b,t,distance)
    return torch.sum(result,dim=1) # [rn, kn]

def generate_hypotheses(coords, direct, vn, num=None, idxs=None):
    assert ((num is not None) and num > 0) or (idxs is not None), "either 'num' or 'idxs' must be given"
    if idxs is None:
        idxs = torch.zeros([num,vn, 2], dtype=torch.long, device=direct.device).random_(0, direct.shape[0])
    else:
        num = idxs.shape[0]
    i0 = idxs[:,:,0]
    i1 = idxs[:,:,1]
    p0 = coords[idxs[:,:,0]] #[rn,vn,2]
    n0 = torch.cat([direct[i0[:,x],x,:].unsqueeze(1) for x in range(vn)],dim=1) #[rn,vn,2]
    p1 = coords[idxs[:,:,1]] #[rn,2]
    n1 = torch.cat([direct[i1[:,x],x,:].unsqueeze(1) for x in range(vn)],dim=1) #[rn,vn,2]
    x,y, valid = intersection_from_dir(p0,p1,n0,n1) #([rn,vn],[rn,vn],[rn,vn])
    cur_hyp_pts = torch.cat([x.unsqueeze(2),y.unsqueeze(2)],dim=2)
    return cur_hyp_pts, valid

class DSACFunction(Function):

    @staticmethod
    def forward(ctx, vertex_in, mask, rounds=16, alpha=0.1, beta=100, tau=0.01):
        ### network output rearrangement
        vertex = vertex_in.permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex.shape
        vn = vn_2//2
        vertex = vertex.view(b, h, w, vn, 2)

        ### generate output tensors
        hp = torch.empty((b, rounds, vn, 2), dtype = vertex.dtype, device = vertex.device)
        hp_scores = torch.empty((b, rounds, vn_2//2), dtype = vertex.dtype, device = vertex.device)
        hp_masks = []
        hp_indexes = []
        hp_valid = [] #[b, rounds, vn]
        ###

        for batch_i in range(b):
            cur_mask = mask[batch_i]
            cur_mask=cur_mask.bool()
            hp_masks.append(cur_mask)

            ### Sample points for random hypothesis generation
            # if too few points, just skip it
            all_coords = torch.nonzero(cur_mask, as_tuple=False).float()
            all_coords = all_coords[:, [1, 0]]
            all_coords_size = all_coords.shape[0]

            if all_coords_size < 5:
                hp_scores[batch_i] = torch.ones_like(hp_scores[batch_i])
                hp[batch_i] = torch.zeros_like(hp[batch_i])
                hp_indexes.append(torch.zeros_like(hp[batch_i]))
                hp_valid.append(torch.zeros_like(hp_scores[batch_i]))
                continue

            all_direct = vertex[batch_i].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3)).view([all_coords_size, vn_2//2, 2])

            idxs = torch.zeros([rounds,vn_2//2, 2], dtype=torch.long, device=mask.device).random_(0, all_direct.shape[0])
            hp_indexes.append(idxs)

            cur_hyp_pts, valid = generate_hypotheses(all_coords, all_direct, vn_2//2, idxs=idxs)
            hp_valid.append(valid)

            ### Score each generated hypothesis
            soft_inliers_batched = similarity_for_hyp_batched(all_coords, all_direct, cur_hyp_pts, beta, tau)
            soft_inliers_batched[valid.logical_not()] = invalid_score
            
            ### Fill output tensors
            hp[batch_i] = cur_hyp_pts
            hp_scores[batch_i] = soft_inliers_batched
            ###
        ctx.save_for_backward(vertex_in, hp, hp_scores)
        ctx.hp_masks = hp_masks
        ctx.hp_indexes = hp_indexes
        ctx.hp_valid = hp_valid
        ctx.b = beta
        ctx.t = tau
        ctx.a = torch.abs(alpha)
        return hp, hp_scores


    @staticmethod
    def backward(ctx, grad_hp, grad_prob):

        '''
        grad_hp = dl/dh : [b, rn, vn, 2]
        grad_prob = dl/dp : [b, rn, vn]

        ds/dv = d/d(vertex) [score]
        dl/dv = d/d(vertex) [loss]
        dl/dh = d/d(hypothesis) [loss] 
        dh/dv = d/d(vertex) [hypothesis]
        
        for each of the vn channels, the gradient of the loss is:
        
        dl/dp * dp/dl + dl/dh * dh/dv =
        dl/dp * p * [ds/vd - E(ds/dv)] + dl/dh * dh_dv

        '''

        grad_vertex = grad_mask = grad_rounds = grad_alpha = grad_beta = grad_tau = None
        if ctx.needs_input_grad[0]:
            vertex_in, hp, hp_scores = ctx.saved_tensors
            vertex = vertex_in.permute(0, 2, 3, 1)
            b, h, w, vn_2 = vertex.shape
            vn = vn_2//2
            vertex = vertex.view(b, h, w, vn, 2)
            
            hp_masks = ctx.hp_masks
            hp_indexes = ctx.hp_indexes
            hp_valid = ctx.hp_valid
            hp_scores = hp_scores.permute(0,2,1) #[b,vn,rn]
            hp_probs = torch.softmax(ctx.a * hp_scores, dim=2) #[b,vn,rn]
            rn = hp_probs.shape[2]
            all_dl_dv = []
            all_rs = torch.LongTensor([i for i in range(rn)])
            for bi in range(b):
                
                cur_mask = hp_masks[bi]
                cur_probs = hp_probs[bi] # [vn, rn]
                cur_valid = hp_valid[bi] # [vn, rn]
                all_coords = torch.nonzero(cur_mask, as_tuple=False).float()
                all_coords = all_coords[:, [1, 0]]
                all_coords_size = all_coords.shape[0]

                if all_coords_size < 5:
                    all_dl_dv.append(torch.zeros_like(vertex[bi]).unsqueeze(0))
                    continue

                all_direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3)).view([all_coords_size, vn_2//2, 2])
                idxs = hp_indexes[bi]
                
                dl_dvh = torch.zeros_like(vertex[bi]).contiguous().detach() #[rn, tn, kn, 2,2]

                #### N.B: We are ignoring the influence of the hypothesis loss in the score loss
                ds_dvh = None #torch.zeros((rn, all_coords_size, vn, 2,2), dtype=vertex.dtype, device=vertex.device).contiguous() #[rn, tn, kn, 2,2]
                i0 = idxs[:,:,0] #[rn,kn]
                i1 = idxs[:,:,1] #[rn,kn]
                p0 = all_coords[idxs[:,:,0]] #[rn,kn,2]
                n0 = torch.cat([all_direct[i0[:,x],x,:].unsqueeze(1) for x in range(vn)],dim=1) #[rn,kn,2]
                p1 = all_coords[idxs[:,:,1]] #[rn,2]
                n1 = torch.cat([all_direct[i1[:,x],x,:].unsqueeze(1) for x in range(vn)],dim=1) #[rn,kn,2]
                dh_dv0,dh_dv1 = dh_dv(p0,p1,n0,n1) #([rn,kn,2,2],[rn,kn,2,2])
                dl_dv0, dl_dv1 = torch.einsum('rkij,rkj->rki',dh_dv0, grad_hp[bi,:,:,:]), torch.einsum('rkij,rkj->rki',dh_dv1, grad_hp[bi,:,:,:]) #[rn, kn, 2]

                # clamp invalid hypotheses losses
                dl_dv0[cur_valid.logical_not(),:] = torch.clamp(dl_dv0[cur_valid.logical_not(),:], -invalid_loss,invalid_loss)
                dl_dv1[cur_valid.logical_not(),:] = torch.clamp(dl_dv1[cur_valid.logical_not(),:], -invalid_loss,invalid_loss)


                hw_index_0 = all_coords[idxs[:,:,0]][:,:,[1,0]].long() #[rn,kn,2]
                hw_index_1 = all_coords[idxs[:,:,1]][:,:,[1,0]].long() #[rn,kn,2]

                for k in range(vn):
                    kappas = torch.ones_like(all_rs) * k 
                    dl_dvh.index_put_((hw_index_0[:, k, 0],hw_index_0[:, k, 1],kappas),dl_dv0[:, k], accumulate=True)
                    dl_dvh.index_put_((hw_index_1[:, k, 0],hw_index_1[:, k, 1],kappas),dl_dv1[:, k], accumulate=True)
                    #ds_dvh.index_put_((all_rs, i0[:, k],kappas), dh_dv0[:, k], accumulate=True)
                    #ds_dvh.index_put_((all_rs, i1[:, k],kappas), dh_dv1[:, k], accumulate=True)

                dl_dvh = torch.clamp(dl_dvh, -kp_loss_clamp, kp_loss_clamp)

                #### sum_j^(rn)( dl/dp * dp/dv) = sum_j^(rn)( dl/dp * pj * dlogp_j/dv) 
                #### dlogpj = a * (dsj/dv - E_j' dsj'/dv)
                dl_dvp = torch.zeros_like(vertex[bi])
                # the 2 here is the contribution of x and y components of vertex
                dhpscores_dvert_all, _, _ = ds_dv(all_coords, all_direct, hp[bi], ctx.a, ctx.b, ctx.t, ds_dvh) #[vn, rn, tn, 2], [vn, rn]
                for k in range(vn):
                    # clamp invalid hypotheses losses
                    dhpscores_dvert_all[k, cur_valid[:,k].logical_not(),:,:] = torch.clamp(dhpscores_dvert_all[k, cur_valid[:,k].logical_not(),:,:],-invalid_loss, invalid_loss )
                
                dhpscores_dvert_exp = torch.sum(dhpscores_dvert_all * cur_probs.unsqueeze(2).unsqueeze(3), dim = 1, keepdim=True) #[vn, 1, tn, 2]
                dp_dvert = ctx.a * (dhpscores_dvert_all - dhpscores_dvert_exp) #[vn, rn, tn, 2]
                dp_dvert = (cur_probs.unsqueeze(2).unsqueeze(3) * dp_dvert) #[vn, rn, tn, 2]
                dl_dvp_dir = (dp_dvert * grad_prob[bi].permute(1,0).unsqueeze(2).unsqueeze(3)).sum(1)  #[vn, tn, 2]
                dl_dvp_dir = torch.clamp(dl_dvp_dir, -score_loss_clamp, score_loss_clamp)
                for k in range(vn):
                    hw_index = all_coords[:,[1,0]].long()
                    dl_dvp[hw_index[:,0],hw_index[:,1],k,:] += dl_dvp_dir[k]
                
                dl_dv_batch =  dl_dvp + dl_dvh 
                
                all_dl_dv.append(dl_dv_batch.unsqueeze(0))
            
            grad_vertex = torch.cat(all_dl_dv,dim=0)
            grad_tau = None
            grad_beta = None
        else:
            print('vertex not requiring grad')

        grad_tau = None
        grad_beta = None

        grad_vertex = grad_vertex.reshape(b,h,w,vn_2).permute(0,3,1,2)

        return grad_vertex, grad_mask, grad_rounds, grad_alpha, grad_beta, grad_tau


class DSACLoss(Function):

    @staticmethod
    def forward(ctx, hp, hp_scores, alpha, gt_kps):
        hp_probs = torch.softmax(alpha*hp_scores, dim=1)
        diffs = hp - gt_kps.unsqueeze(1)
        norm_diffs = torch.sqrt(torch.pow(diffs,2).sum(3)) # [b, rn, vn]
        expected_loss = (norm_diffs * hp_probs).sum(1)
        ctx.save_for_backward(hp, hp_probs, gt_kps)
        return expected_loss.mean()

    
    @staticmethod
    def backward(ctx, grad_loss):
        '''
         ds/dv = d/d(vertex) [score]
         dl/dv = d/d(vertex) [loss]
         dl/dh = d/d(hypothesis) [loss] 
         dh/dv = d/d(vertex) [hypothesis]
        
         for each of the vn channels, the gradient of the loss is:
        
           dl/dh = p * (h-h*)/||h-h*||
           dl/dp = ||h-h*||
        
        '''

        grad_hp = grad_hp_scores = grad_alpha = grad_gt_kps = None

        hp, hp_probs, gt_kps = ctx.saved_tensors
        diffs = hp - gt_kps.unsqueeze(1) # [b,rn,vn,2]
        norm_diffs = torch.sqrt(torch.pow(diffs,2).sum(3)) # [b, rn, vn]

        grad_hp = hp_probs.unsqueeze(3) * (diffs/norm_diffs.unsqueeze(3))
        grad_hp_scores = norm_diffs # torch.clamp_max(norm_diffs, 10)

        grad_hp = grad_hp * grad_loss
        grad_hp_scores = grad_hp_scores * grad_loss

        return grad_hp, grad_hp_scores, grad_alpha, grad_gt_kps
