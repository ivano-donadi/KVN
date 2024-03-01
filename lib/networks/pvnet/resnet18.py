from importlib.metadata import requires
from locale import currency
from tkinter import E
from torch import nn
import torch
from torch.nn import functional as F
from .resnet import resnet18
from .dsac_layer import DSACFunction
import lib.csrc.ransac_voting.ransac_voting as ransac_voting
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from lib.csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer, ransac_voting_layer_v3, estimate_voting_distribution_with_mean, ransac_mask_make_unimodal
from .pvnet_head import PVNetHead

# from https://github.com/artste/fastai-samples/blob/master/kaggle/lesson2-protein-human-protein-atlas-384-resnet50_data_block.ipynb
def make_batches(x,bs):
    '''
    Sample make_batches(11,3) = [3,3,3,2]
    '''
    if(x<=bs):
        return [min(x,bs)]
    else:
        return [bs] + make_batches(x-bs,bs)

# from https://github.com/artste/fastai-samples/blob/master/kaggle/lesson2-protein-human-protein-atlas-384-resnet50_data_block.ipynb
def create_new_weights(original_weights,nChannels):
    dst = torch.zeros(64,nChannels,7,7)
    #Repeat original weights up to fill dimension
    start=0
    for i in make_batches(nChannels,3):
        #print('dst',start,start+i, ' = src',0,i)
        dst[:,start:start+i,:,:]=original_weights[:,:i,:,:]
        start = start+i
    return dst


# from https://github.com/artste/fastai-samples/blob/master/kaggle/lesson2-protein-human-protein-atlas-384-resnet50_data_block.ipynb
def adapt_first_layer(src_model, nChannels):
    '''
    Change first layer of network to accomodate new channels
    '''

    # TODO: check if it worked 

    # save original
    original_weights = src_model.weight.clone()

    # create new repeating existent weights channelwise
    new_weights = create_new_weights(original_weights,nChannels)

    # create new layes
    new_layer = nn.Conv2d(nChannels,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    new_layer.weight = nn.Parameter(new_weights)

    return new_layer

def normalize_vertex(vertex):
    vertex = vertex.permute(0, 2, 3, 1)
    b, h, w, vn_2 = vertex.shape
    vn = vn_2//2
    vertex = vertex.view(b, h, w, vn, 2)
    vertex = nn.functional.normalize(vertex, p=2, dim = 4)
    vertex = vertex.view(b, h, w, vn_2)
    vertex = vertex.permute(0,3,1,2)
    return vertex

class Resnet18(nn.Module):
    def __init__(self, ver_dim, seg_dim, train_cfg, test_ransac_rounds,fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(Resnet18, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        self.ver_dim = ver_dim
        self.seg_dim = seg_dim

        self.vote_head = PVNetHead(ver_dim, fcdim, s8dim, s4dim, s2dim, raw_dim)
        self.seg_head = PVNetHead(seg_dim, fcdim, s8dim, s4dim, s2dim, raw_dim)

        self.use_dsac = train_cfg.use_dsac
        self.softmax_a = nn.Parameter(torch.ones(1)*(train_cfg.dsac_alpha))
        self.softmax_b = nn.Parameter(torch.ones(1)*train_cfg.dsac_beta)
        self.softmax_t = nn.Parameter(torch.ones(1)*train_cfg.dsac_threshold)
        self.dsac_layer = DSACFunction.apply
        self.ransac_rounds = train_cfg.ransac_rounds
        self.test_ransac_rounds=test_ransac_rounds


    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def decode_keypoint(self, output):
        for suffix in ['_L','_R']:
            vertex = output['vertex'+suffix].permute(0, 2, 3, 1)
            b, h, w, vn_2 = vertex.shape
            vertex = vertex.view(b, h, w, vn_2//2, 2)
            mask = torch.argmax(output['seg'+suffix], 1)
            mean = ransac_voting_layer_v3(mask, vertex, self.ransac_rounds, inlier_thresh=0.99)
            winning_kps, cov = estimate_voting_distribution_with_mean(mask, vertex, mean,round_hyp_num=self.ransac_rounds, min_hyp_num=self.test_ransac_rounds)
            output.update({'mask'+suffix: mask, 'kpt_2d'+suffix: winning_kps, 'aux_mask'+suffix: mask, 'var'+suffix:cov})
        
    def forward(self, batch, epoch=0, feature_alignment=False):
        input_L, input_R = (batch['inp_L'],batch['inp_R'])

        features_L = self.resnet18_8s(input_L)
        seg_pred_L=self.seg_head(features_L, input_L)
        ver_pred_L=self.vote_head(features_L, input_L,False)
        mask_L = torch.argmax(seg_pred_L, 1)

        features_R = self.resnet18_8s(input_R)
        seg_pred_R =self.seg_head(features_R, input_R)
        ver_pred_R =self.vote_head(features_R, input_R, False)
        mask_R = torch.argmax(seg_pred_R, 1)

        ### A normalized vertex prediction is an essential assumption inside DSAC, if you
        ### remove these two lines you need to change the dsac backward to take the norm into account 
        ver_pred_L = normalize_vertex(ver_pred_L)
        ver_pred_R = normalize_vertex(ver_pred_R)


        if self.use_dsac:
            # avoid focusing on wrong pixels during training
            if self.training:
                dsac_mask_L = batch['mask_L']
                dsac_mask_R = batch['mask_R']
            else:
                dsac_mask_L = mask_L
                dsac_mask_R = mask_R
            hp_L, hp_scores_L = self.dsac_layer(ver_pred_L, dsac_mask_L, self.ransac_rounds, self.softmax_a, self.softmax_b, self.softmax_t) 
            hp_R, hp_scores_R = self.dsac_layer(ver_pred_R, dsac_mask_R, self.ransac_rounds, self.softmax_a, self.softmax_b, self.softmax_t) 
        else:
            hp_L = hp_R = hp_scores_L = hp_scores_R = None

        ret = {'seg_L': seg_pred_L, 'vertex_L': ver_pred_L, 'keypoints_L': hp_L,'scores_L': hp_scores_L,
               'seg_R': seg_pred_R, 'vertex_R': ver_pred_R, 'keypoints_R': hp_R,'scores_R': hp_scores_R}

        if not self.training:
            with torch.no_grad():
                self.decode_keypoint(ret)

        return ret
    
    def dist_from_scores(self, scores, detached=False, log=False):
        if detached:
            a = torch.abs(self.softmax_a).detach()
        else:
            a = torch.abs(self.softmax_a)
        s = a * scores
        if log:
            return torch.log_softmax(s, dim=1)
        else:
            return torch.softmax(s, dim=1)
        

def get_res_pvnet(ver_dim, seg_dim, train_cfg, test_ransac_rounds):
    model = Resnet18(ver_dim, seg_dim, train_cfg, test_ransac_rounds=test_ransac_rounds)
    return model


