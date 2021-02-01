#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:10:49 2021

@author: quentin
"""

import os
import numpy as np
import torch
import argparse
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader import Dataset, default_loader
from new_func import reshape
from util import *
import scipy.misc
import time
from types import SimpleNamespace


def styleTransferBarycenter(contentImg,styleImgs,csF,alphas, method=["WCT","WCT","WCT","WCT","WCT"], device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    vgg1 = 'models/vgg_normalised_conv1_1.t7'
    vgg2 = 'models/vgg_normalised_conv2_1.t7'
    vgg3 = 'models/vgg_normalised_conv3_1.t7'
    vgg4 = 'models/vgg_normalised_conv4_1.t7'
    vgg5 = 'models/vgg_normalised_conv5_1.t7'

    decoder1 = 'models/feature_invertor_conv1_1.t7'
    decoder2 = 'models/feature_invertor_conv2_1.t7'
    decoder3 = 'models/feature_invertor_conv3_1.t7'
    decoder4 = 'models/feature_invertor_conv4_1.t7'
    decoder5 = 'models/feature_invertor_conv5_1.t7'

    args = SimpleNamespace(vgg1=vgg1, vgg2=vgg2, vgg3=vgg3, vgg4=vgg4, vgg5=vgg5, decoder1=decoder1, decoder2=decoder2, decoder3=decoder3, decoder4=decoder4, decoder5=decoder5)

    wct = WCT(args).to(device)

    cF5 = wct.e5(contentImg)
    cF5 = cF5.data.cpu().squeeze(0)
    sF5s = [wct.e5(styleImg) for styleImg in styleImgs]
    sF5s = [sF5.data.cpu().squeeze(0) for sF5 in sF5s]
    csF5 = wct.transformBarycenter(cF5,sF5s,csF,alphas[0],method=method[0],n_iter=5000)
    Im5 = wct.d5(csF5)

    cF4 = wct.e4(Im5)
    cF4 = cF4.data.cpu().squeeze(0)
    sF4s = [wct.e4(styleImg) for styleImg in styleImgs]
    sF4s = [sF4.data.cpu().squeeze(0) for sF4 in sF4s]
    csF4 = wct.transformBarycenter(cF4,sF4s,csF,alphas[1],method=method[1],n_iter=3000)
    Im4 = wct.d4(csF4)

    sF3s = [wct.e3(styleImg) for styleImg in styleImgs]
    cF3 = wct.e3(Im4)
    sF3s = [sF3.data.cpu().squeeze(0) for sF3 in sF3s]
    cF3 = cF3.data.cpu().squeeze(0)
    csF3 = wct.transformBarycenter(cF3,sF3s,csF,alphas[2],method=method[2],n_iter=2000)
    Im3 = wct.d3(csF3)

    sF2s = [wct.e2(styleImg) for styleImg in styleImgs]
    cF2 = wct.e2(Im3)
    sF2s = [sF2.data.cpu().squeeze(0) for sF2 in sF2s]
    cF2 = cF2.data.cpu().squeeze(0)
    csF2 = wct.transformBarycenter(cF2,sF2s,csF,alphas[3],method=method[3],n_iter=300)
    Im2 = wct.d2(csF2)

    cF1 = wct.e1(Im2)
    cF1 = cF1.data.cpu().squeeze(0)
    sF1s = [wct.e1(styleImg) for styleImg in styleImgs]
    sF1s = [sF1.data.cpu().squeeze(0) for sF1 in sF1s]
    csF1 = wct.transformBarycenter(cF1,sF1s,csF,alphas[4],method=method[4],n_iter=100)
    Im1 = wct.d1(csF1)
    # save_image has this wired design to pad images with 4 pixels at default.
    #vutils.save_image(Im1.data.cpu().float(),os.path.join(args.outf,imname))
    return Im1[0].cpu().detach().permute(1,2,0).numpy()

def style_barycenter(content,styles,resize_content=512,resize_style=512,alphas=None, method=["WCT","WCT","WCT","WCT","WCT"],device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    if alphas is None
        # set alpha uniformly
        alphas = np.array([[1 / (len(styles) + 1) for _ in styles] for _ in range(5)])

    content = default_loader(content)
    styles = [default_loader(style) for style in styles]

    if resize_content!=0:
      content = reshape(content,resize_content)
    if resize_style!=0:
      styles = [reshape(style,resize_style) for style in styles]

    content = transforms.ToTensor()(content)
    styles = [transforms.ToTensor()(style) for style in styles]

    avgTime = 0
    #cImg = torch.Tensor().to(device)
    #sImgs = [torch.Tensor().to(device) for _ in styles]
    csF = torch.Tensor().to(device)
    csF = Variable(csF).to(device)

    cImg = Variable(content[None,:],volatile=True).to(device)
    sImgs = [Variable(style[None,:],volatile=True).to(device) for style in styles]
    start_time = time.time()
    # WCT Style Transfer
    image = styleTransferBarycenter(cImg,sImgs,csF,alphas, device=device, method=method)

    end_time = time.time()
    print('Elapsed time is: %f' % (end_time - start_time))

    return image