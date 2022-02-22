from turtle import window_width
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models # add models to the list
import os

import argparse
parser = argparse.ArgumentParser(description='Train-Test-Deploy')
parser.add_argument('mode', type=str, default="train",
                    help='train-test, test or deploy')
args = parser.parse_args()

train_set = ['../datasets/CISL24218/',]
train_sample_rate = None
test_set = ['../datasets/NISTSD27/',]
deploy_set = ['../datasets/NISTSD27/images/','../datasets/CISL24218/', \
            '../datasets/FVC2002DB2A/','../datasets/NIST4/','../datasets/NIST14/']

output_dir = '../output/'+datetime.now().strftime('%Y%m%d-%H%M%S')

#python train_test_deploy.py deploy


def img_normalization(im_input, m0 = 0.0, var0 = 1.0):
    m = np.mean(im_input.numpy())
    var = np.var(im_input.numpy())
    im_input = im_input.apply_(lambda x: (m0 + (np.sqrt((var0*(x-m)*(x-m))/var))) if (x>m) else (m0-(np.sqrt((var0*(x-m)*(x-m))/var))))
    return im_input         

def orientation(image, stride = 8, window = 17):
    assert image.shape[1] == 1, 'Images must be Grayscale'
    E = torch.FloatTensor(np.reshape(np.ones([window,window,1,1]),[1,1,window,window]))
    sobelx = torch.FloatTensor(np.reshape(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float), [1,1,3,3]))
    sobely = torch.FloatTensor(np.reshape(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float), [1,1,3,3]))
    gaussian_mask = torch.FloatTensor(np.reshape(np.array([[ 0.0000, 0.0000, 0.0002, 0.0000, 0.0000 ], \
                                                            [ 0.0000, 0.0113, 0.0837, 0.0113, 0.0000 ], \
                                                            [ 0.0002, 0.0837, 0.6187, 0.0837, 0.0002 ], \
                                                            [ 0.0000, 0.0113, 0.0837, 0.0113, 0.0000 ], \
                                                            [ 0.0000, 0.0000, 0.0002, 0.0000, 0.0000 ]]),[1,1,5,5]))
    #sobel_gradient
    Ix = F.conv2d(image,sobelx,stride=1,padding='same')
    Iy = F.conv2d(image,sobely,stride=1,padding='same')
    #elt_wise1
    Ix2 = torch.mm(Ix.view(512,512),Ix.view(512,512))
    Ix2 = Ix2.view(1,1,512,512)
    Iy2 = torch.mm(Iy.view(512,512),Iy.view(512,512))
    Iy2 = Iy2.view(1,1,512,512)    
    Ixy = torch.mm(Ix.view(512,512),Iy.view(512,512))
    Ixy = Ixy.view(1,1,512,512)
    #range_sum
    Gxx = F.conv2d(Ix2,E,stride=1,padding='same')
    Gyy = F.conv2d(Iy2,E,stride=1,padding='same')
    Gxy = F.conv2d(Ixy,E,stride=1,padding='same')
    #eltwise_2
    Gxx_Gyy = Gxx.sub(Gyy)
    theta = torch.atan2((2*Gxy),Gxx_Gyy) + np.pi        
    #gaussian_filter
    phi_x = F.conv2d(np.cos(image),gaussian_mask,stride=1,padding='same')
    phi_y = F.conv2d(np.sin(image),gaussian_mask,stride=1,padding='same')
    theta = torch.atan2(phi_y,phi_x)/2
    return theta
    
def get_tra_ori():
    img_input=Input(shape=(None, None, 1))
    theta = Lambda(orientation)(img_input)
    model = Model(inputs=[img_input,], outputs=[theta,])
    return model
tra_ori_model = get_tra_ori()


def main():
    if args.mode == 'train':
        train()
    elif args.mode == 'test':        
        for folder in test_set:
            test([folder,], pretrain, output_dir+"/", test_num=258, draw=False) 
    elif args.mode == 'deploy':
        for i, folder in enumerate(deploy_set):
            deploy(folder, str(i))
    else:
        pass

if __name__ =='__main__':
    main()
