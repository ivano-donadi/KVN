import torch.nn as nn
import torch
from lib.networks.pvnet.dsac_layer import DSACLoss
from lib.utils.pvnet.visualize_utils import visualize_dsac_results, visualize_dsac_results_split


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_cfg):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.vote_crit = torch.nn.functional.smooth_l1_loss
        self.seg_crit = nn.CrossEntropyLoss()
        self.prv_loss_stats = None
        self.kp_threshold = train_cfg.kp_loss_threshold
        self.kp_beta = train_cfg.kp_loss_beta
        self.target_entropy = train_cfg.target_entropy
        self.soft_ap_beta = train_cfg.soft_ap_beta
        self.use_epipolar = train_cfg.use_epipolar
        self.use_dsac = train_cfg.use_dsac

    def forward(self, batch, epoch):
        output = self.net(batch, epoch=epoch)


        scalar_stats = {}

        vote_loss = 0
        kp_loss = 0
        seg_loss = 0
        entropy_loss = 0


        for suffix in ['_L','_R']:
            
            weight = batch['mask'+suffix][:, None].float()
            curr_vote_loss = self.vote_crit(output['vertex'+suffix] * weight, batch['vertex'+suffix] * weight, reduction='sum')
            if weight.sum() > 0:
                curr_vote_loss = curr_vote_loss / weight.sum() / batch['vertex'+suffix].size(1)
            vote_loss += curr_vote_loss

            mask = batch['mask'+suffix].long()
            curr_seg_loss = self.seg_crit(output['seg'+suffix], mask)
            seg_loss += curr_seg_loss

            if self.use_dsac:
                all_pred_kps = output['keypoints'+suffix]
                pred_scores = output['scores'+suffix]
                gt_kps = batch['kpt_2d'+suffix]
                # do not let kp loss influence softmax weight
                curr_kp_loss = DSACLoss.apply(all_pred_kps, pred_scores, self.net.softmax_a, gt_kps)
                kp_loss += curr_kp_loss
                
                # do not let entropy loss influence anything other than the softmax weight
                entropy_loss += self.entropy_loss_function(output, suffix)

        vote_loss = vote_loss / 2
        seg_loss = seg_loss / 2

        if self.use_dsac:
            kp_loss = (kp_loss / 2) 
            entropy_loss = entropy_loss / 2

            kp_weight_criterion = self.kp_beta * (self.kp_threshold - kp_loss.detach())
            kp_ext_weight = 1./(1+torch.exp(-1*kp_weight_criterion))
            vote_ext_weight = 1. - kp_ext_weight

            scalar_stats.update({'kp_ext_weight': torch.FloatTensor([kp_ext_weight]), 'vote_ext_weight': torch.FloatTensor([vote_ext_weight]), 'softmax_a': self.net.softmax_a, 'softmax_b': self.net.softmax_b, 'softmax_t': self.net.softmax_t})
            if self.use_epipolar and epoch > 49:
                epipolar_loss = self.epipolar_loss_function(batch, output)
                vertex_loss = kp_ext_weight * (kp_loss+epipolar_loss) + vote_ext_weight * vote_loss
            else:
                vertex_loss = kp_ext_weight * kp_loss + vote_ext_weight * vote_loss
        else:
            vertex_loss = vote_loss
        
        loss = vertex_loss + entropy_loss + seg_loss

        #visualize_dsac_results_split(batch['inp_L'][0], output['keypoints_L'][0], self.net.dist_from_scores(output['scores_L'])[0], batch['kpt_2d_L'][0].detach().cpu())
        scalar_stats.update({'vote_loss': vote_loss.detach().cpu(),  'seg_loss': seg_loss.detach().cpu(),  'vertex_loss': vertex_loss.detach().cpu()})
        if self.use_dsac:
            scalar_stats.update({'entropy_loss': entropy_loss.detach().cpu(), 'kp_loss': kp_loss.detach().cpu()})
            if self.use_epipolar  and epoch > 49:
                scalar_stats.update({'epipolar_loss': epipolar_loss.detach().cpu()})

        scalar_stats.update({'loss': loss.detach().cpu()})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

    def update_loss_stats(self,loss_stats):
        self.prv_loss_stats = loss_stats

    def entropy_loss_function(self, output, suffix):
        pred_scores = output['scores'+suffix]
        b = pred_scores.shape[0]
        det_probs = self.net.dist_from_scores(pred_scores.detach(), False)
        det_log_probs = self.net.dist_from_scores(pred_scores.detach(), False, True)/torch.log(torch.ones(1)*2).cuda()
        target_entropy = torch.ones(1,device=pred_scores.device, dtype=pred_scores.dtype) * self.target_entropy
        entropy = (-1 * (det_probs*det_log_probs).sum(1)).mean(1) # [b]
        curr_entropy_loss = torch.abs(entropy - target_entropy[None,:]).sum() / b
        return curr_entropy_loss

    def epipolar_loss_function(self, batch, output):
        pred_kp_L = output['keypoints_L'].detach() + batch['offset_L'][:,None,None,:] # b, rn, vn, 2
        pred_scores_L = output['scores_L'] # b, rn ,vn
        probs_L = self.net.dist_from_scores(pred_scores_L, True) # b, rn, vn
        pred_kp_R = output['keypoints_R'].detach() + batch['offset_R'][:,None,None,:] # b, rn, vn, 2
        pred_scores_R = output['scores_R'] # b, rn ,vn
        probs_R = self.net.dist_from_scores(pred_scores_R, True)

        epipolar_loss = 0
        for b in range(pred_kp_L.shape[0]):
            winners_index_L = torch.multinomial(probs_L[b].permute(1,0), 1).squeeze(1) # vn
            winners_index_R = torch.multinomial(probs_R[b].permute(1,0), 1).squeeze(1) # vn
            b_epipolar_loss = 0
            for k in range(pred_kp_L.shape[2]):
                y_L = pred_kp_L[b, winners_index_L[k], k, 1] # 1
                y_R = pred_kp_R[b, winners_index_R[k], k, 1] # 1
                diff = (y_L - y_R).pow(2).sqrt()
                if diff.isnan() or diff.isinf():
                    diff = 0
                b_epipolar_loss = b_epipolar_loss + diff
            b_epipolar_loss = b_epipolar_loss/pred_kp_L.shape[2]
            epipolar_loss = epipolar_loss + b_epipolar_loss
        epipolar_loss = epipolar_loss/pred_kp_L.shape[0]

        return epipolar_loss
