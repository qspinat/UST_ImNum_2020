#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:10:49 2021

@author: quentin
"""

import os
import torch
import argparse
from PIL import Image
import torch
import numpy as np
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader import Dataset, default_loader
from util import *
import scipy.misc
import time
from types import SimpleNamespace



def styleTransfer(contentImg,styleImg,csF,alpha=0.5, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

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

    contentPath = 'images/content'
    stylePath = 'images/style'

    args = SimpleNamespace(vgg1=vgg1, vgg2=vgg2, vgg3=vgg3, vgg4=vgg4, vgg5=vgg5, decoder1=decoder1, decoder2=decoder2, decoder3=decoder3, decoder4=decoder4, decoder5=decoder5)

    wct = WCT(args).to(device)
    
    if len(np.shape(alpha))==0:
        alpha = [alpha,alpha,alpha,alpha,alpha]

    sF5 = wct.e5(styleImg)
    cF5 = wct.e5(contentImg)
    sF5 = sF5.data.cpu().squeeze(0)
    cF5 = cF5.data.cpu().squeeze(0)
    csF5 = wct.transform(cF5,sF5,csF,alpha[0])
    Im5 = wct.d5(csF5)

    sF4 = wct.e4(styleImg)
    cF4 = wct.e4(Im5)
    sF4 = sF4.data.cpu().squeeze(0)
    cF4 = cF4.data.cpu().squeeze(0)
    csF4 = wct.transform(cF4,sF4,csF,alpha[1])
    Im4 = wct.d4(csF4)

    sF3 = wct.e3(styleImg)
    cF3 = wct.e3(Im4)
    sF3 = sF3.data.cpu().squeeze(0)
    cF3 = cF3.data.cpu().squeeze(0)
    csF3 = wct.transform(cF3,sF3,csF,alpha[2])
    Im3 = wct.d3(csF3)

    sF2 = wct.e2(styleImg)
    cF2 = wct.e2(Im3)
    sF2 = sF2.data.cpu().squeeze(0)
    cF2 = cF2.data.cpu().squeeze(0)
    csF2 = wct.transform(cF2,sF2,csF,alpha[3])
    Im2 = wct.d2(csF2)

    sF1 = wct.e1(styleImg)
    cF1 = wct.e1(Im2)
    sF1 = sF1.data.cpu().squeeze(0)
    cF1 = cF1.data.cpu().squeeze(0)
    csF1 = wct.transform(cF1,sF1,csF,alpha[4])
    Im1 = wct.d1(csF1)
    # save_image has this wired design to pad images with 4 pixels at default.
    #vutils.save_image(Im1.data.cpu().float(),os.path.join(args.outf,imname))
    return Im1[0].cpu().detach().permute(1,2,0).numpy()

def reshape(img, fineSize):
    w,h = img.size
    if (w > h):
        if (w != fineSize):
            neww = fineSize
            newh = int(h*neww/w)
            img = img.resize((neww,newh))
    else:
        if (h != fineSize):
            newh = fineSize
            neww = int(w*newh/h)
            img = img.resize((neww,newh))
    return img

def easy_transfert(content,style,resize=512,alpha=0.5,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    content = default_loader(content)
    style = default_loader(style)

    content = reshape(content,resize)
    style = reshape(style,resize)

    content = transforms.ToTensor()(content)
    style = transforms.ToTensor()(style)

    avgTime = 0
    cImg = torch.Tensor().to(device)
    sImg = torch.Tensor().to(device)
    csF = torch.Tensor().to(device)
    csF = Variable(csF).to(device)

    cImg = Variable(content[None,:],volatile=True).to(device)
    sImg = Variable(style[None,:],volatile=True).to(device)
    start_time = time.time()
    # WCT Style Transfer

    image = styleTransfer(cImg,sImg,csF,alpha=alpha)

    end_time = time.time()
    print('Elapsed time is: %f' % (end_time - start_time))
    avgTime += (end_time - start_time)

    print('Averaged time is %f' % (avgTime))

    return image