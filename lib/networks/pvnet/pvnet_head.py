import torch
import torch.nn as nn

class PVNetHead(nn.Module):
    def __init__(self, output_channels,fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(PVNetHead, self).__init__()

        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(128+fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(64+s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )

        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(raw_dim, output_channels, 1, 1)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, features, input, detached = False):
        x2s, x4s, x8s, _, _, xfc = features

        if detached:
            x2s = x2s.detach()
            x4s = x4s.detach()
            x8s = x8s.detach()
            xfc = xfc.detach()


        fm=self.conv8s(torch.cat([xfc,x8s],1))
        fm=self.up8sto4s(fm)
        
        if fm.shape[2]==136:
            fm = nn.functional.interpolate(fm, (135,180), mode='bilinear', align_corners=False)
        fm=self.conv4s(torch.cat([fm,x4s],1))
        fm=self.up4sto2s(fm)
        if fm.shape[2]==64:
            fm = nn.functional.interpolate(fm, (63,112), mode='bilinear', align_corners=False)

        fm=self.conv2s(torch.cat([fm,x2s],1))
        fm=self.up2storaw(fm)
        x=self.convraw(torch.cat([fm,input],1))

        return x
