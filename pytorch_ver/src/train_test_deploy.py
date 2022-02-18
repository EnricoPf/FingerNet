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

#python train_test_deploy.py deploy


def img_normalization(img_input, m0=0.0, v0=1.0):
    transform = transforms.compose([
        transforms.toTensor()
        transforms.Normalize(mean = m,std = sqrt(var))
    ])
    np_image = img_input.numpy()
    m = np.mean(np_image)
    std = np.std(np_image)
    
    
    return;



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
