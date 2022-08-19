

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 23:05:23 2020

@author: zhang
"""

import torch.nn as nn
import torch
from functools import partial
import torch.nn.functional as F
import math
from sobel import Gedge_map
from sobel import edge_conv2d128
from sobel import edge_conv2d256
from vit_seg_configs import get_b16_config
from vit_seg_modeling import VisionTransformer

nonlinearity = partial(F.relu, inplace=True)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = residual + out
        out = self.relu(out)
        return out
    

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
        
class up_conv8(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv8, self).__init__()
        self.upsam = nn.Upsample(scale_factor=2)
        self.upconv1 = conv_block(ch_in, 256)
        self.upconv2 = conv_block(256, 128)
        self.upconv3 = conv_block(128, ch_out)

    def forward(self, x):
        x = self.upsam(x)
        x = self.upconv1(x)
        x = self.upsam(x)
        x = self.upconv2(x)
        x = self.upsam(x)
        x = self.upconv3(x)
        return x
    
        
class up_conv4(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv4, self).__init__()
        self.upsam = nn.Upsample(scale_factor=2)
        self.upconv1 = conv_block(ch_in, 128)
        self.upconv2 = conv_block(128, ch_out)

    def forward(self, x):
        x = self.upsam(x)
        x = self.upconv1(x)
        x = self.upsam(x)
        x = self.upconv2(x)
        return x


class local_attention(nn.Module):
    def __init__(self, channel):
        super(local_attention, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel // 2, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel//2, channel // 2, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel//2, channel // 2, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel // 2, 1, kernel_size=1, dilation=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.u1 = torch.nn.Parameter(torch.ones((1,1), dtype = torch.float32))
        self.u2 = torch.nn.Parameter(torch.ones((1,1), dtype = torch.float32))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        self.psi = nn.Sequential(
            nn.Conv2d(channel // 2, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 2, channel, bias=False),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)


        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, H, W = x.size()
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(self.dilate1(x)))
        dilate3_out = nonlinearity(self.dilate3(self.dilate2(self.dilate1(x))))

        fea1 = dilate1_out
        fea2 = dilate2_out
        fea3 = dilate3_out

        fea = fea1+fea2+fea3

        edgemap = self.relu(Gedge_map(self.psi(fea))+self.psi(fea))

        x = x*edgemap
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)+x

class shallow_fea_fusion(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(shallow_fea_fusion,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)


        self.upsam = up_conv(ch_in=128, ch_out=64)
        self.shallow_conv = conv_block(ch_in=128, ch_out=64)
        self.conv1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self,g,x):
        # 下采样的gating signal 卷积
        g = self.upsam(g)
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1+x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        fea1 = x*psi
        fea2 = g*psi
        fea = torch.cat((fea1,fea2),dim=1)
        fea = self.shallow_conv(fea)
        fea = self.conv1x1(fea)
        return fea

class U_Net(nn.Module):
    def __init__(self, in_c, n_classes):
        super(U_Net, self).__init__()
        self.n_classes = n_classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_c, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.l_at4 = local_attention(256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.l_at3 = local_attention(128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.l_at2 = local_attention(64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
                      
        self.Up8x8 = up_conv8(ch_in=512, ch_out=1)
        self.Up4x4 = up_conv4(ch_in=256, ch_out=1)

        self.fconv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)

        self.shallow_fusion = shallow_fea_fusion(F_g=64,F_l=64,F_int=64)
        self.Trans = VisionTransformer(get_b16_config())

    def forward(self, x):
        x1 = self.Conv1(x) 

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Trans(x3)


        d4 = self.Up4(x4)
        lt2 = self.l_at4(x3)
        x3 = lt2
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        lt3 = self.l_at3(x2)
        x2 = lt3
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        lt4 = self.l_at2(x1)
        x1 = lt4
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        m4 = self.Up8x8(x4)
        m3 = self.Up4x4(d4) 
        
        deep_fea = m4+m3

        shallow_fea = self.shallow_fusion(d3,d2)
        out = self.fconv(deep_fea+shallow_fea)

        return out