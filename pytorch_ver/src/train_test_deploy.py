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
    strides = [1,stride,stride,1]
    E = torch.FloatTensor(np.ones([1,1,window,window]))
    sobelx = torch.FloatTensor(np.reshape(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float), [1,1,3,3]))
    sobely = torch.FloatTensor(np.reshape(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float), [1,1,3,3]))
    gaussian_mask = torch.FloatTensor(np.reshape(np.array([[ 0.0000, 0.0000, 0.0002, 0.0000, 0.0000 ], [ 0.0000, 0.0113, 0.0837, 0.0113, 0.0000 ], \
                              [ 0.0002, 0.0837, 0.6187, 0.0837, 0.0002 ], [ 0.0000, 0.0113, 0.0837, 0.0113, 0.0000 ], \
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
    
class get_tra_ori(nn.Module):
    def forward(self, input):
        return orientation(input)
tra_ori_model = get_tra_ori()

def sub_load_data(data, img_size, aug): 
    img_name, dataset = data
    img = misc.imread(dataset+'images/'+img_name+'.bmp', mode='L')
    seg = misc.imread(dataset+'seg_labels/'+img_name+'.png', mode='L')
    try:
        ali = misc.imread(dataset+'ori_labels/'+img_name+'.bmp', mode='L')
    except:
        ali = np.zeros_like(img)
    mnt = np.array(mnt_reader(dataset+'mnt_labels/'+img_name+'.mnt'), dtype=float)
    if any(img.shape != img_size):
        # random pad mean values to reach required shape
        if np.random.rand()<aug:
            tra = np.int32(np.random.rand(2)*(np.array(img_size)-np.array(img.shape)))
        else:
            tra = np.int32(0.5*(np.array(img_size)-np.array(img.shape)))
        img_t = np.ones(img_size)*np.mean(img)
        seg_t = np.zeros(img_size)
        ali_t = np.ones(img_size)*np.mean(ali)
        img_t[tra[0]:tra[0]+img.shape[0],tra[1]:tra[1]+img.shape[1]] = img
        seg_t[tra[0]:tra[0]+img.shape[0],tra[1]:tra[1]+img.shape[1]] = seg
        ali_t[tra[0]:tra[0]+img.shape[0],tra[1]:tra[1]+img.shape[1]] = ali
        img = img_t
        seg = seg_t
        ali = ali_t
        mnt = mnt+np.array([tra[1],tra[0],0]) 
    if np.random.rand()<aug:
        # random rotation [0 - 360] & translation img_size / 4
        rot = np.random.rand() * 360
        tra = (np.random.rand(2)-0.5) / 2 * img_size 
        img = ndimage.rotate(img, rot, reshape=False, mode='reflect')
        img = ndimage.shift(img, tra, mode='reflect')
        seg = ndimage.rotate(seg, rot, reshape=False, mode='constant')
        seg = ndimage.shift(seg, tra, mode='constant')
        ali = ndimage.rotate(ali, rot, reshape=False, mode='reflect')
        ali = ndimage.shift(ali, tra, mode='reflect') 
        mnt_r = point_rot(mnt[:, :2], rot/180*np.pi, img.shape, img.shape)  
        mnt = np.column_stack((mnt_r+tra[[1, 0]], mnt[:, 2]-rot/180*np.pi))
    # only keep mnt that stay in pic & not on border
    mnt = mnt[(8<=mnt[:,0])*(mnt[:,0]<img_size[1]-8)*(8<=mnt[:, 1])*(mnt[:,1]<img_size[0]-8), :]
    return img, seg, ali, mnt   

def load_data(dataset, tra_ori_model, rand=False, aug=0.0, batch_size=1, sample_rate=None):
    if type(dataset[0]) == str:
        img_name, folder_name, img_size = get_maximum_img_size_and_names(dataset, sample_rate)
    else:
        img_name, folder_name, img_size = dataset
    if rand:
        rand_idx = np.arange(len(img_name))
        np.random.shuffle(rand_idx)
        img_name = img_name[rand_idx]
        folder_name = folder_name[rand_idx]
        
    if batch_size > 1 and use_multiprocessing==True:
        p = Pool(batch_size)        
    p_sub_load_data = partial(sub_load_data, img_size=img_size, aug=aug)
    for i in xrange(0,len(img_name), batch_size):
        have_alignment = np.ones([batch_size, 1, 1, 1])
        image = np.zeros((batch_size, img_size[0], img_size[1], 1))
        segment = np.zeros((batch_size, img_size[0], img_size[1], 1))
        alignment = np.zeros((batch_size, img_size[0], img_size[1], 1))
        minutiae_w = np.zeros((batch_size, img_size[0]/8, img_size[1]/8, 1))-1
        minutiae_h = np.zeros((batch_size, img_size[0]/8, img_size[1]/8, 1))-1
        minutiae_o = np.zeros((batch_size, img_size[0]/8, img_size[1]/8, 1))-1
        batch_name = [img_name[(i+j)%len(img_name)] for j in xrange(batch_size)]
        batch_f_name = [folder_name[(i+j)%len(img_name)] for j in xrange(batch_size)]
        if batch_size > 1 and use_multiprocessing==True:    
            results = p.map(p_sub_load_data, zip(batch_name, batch_f_name))
        else:
            results = map(p_sub_load_data, zip(batch_name, batch_f_name))
        for j in xrange(batch_size):
            img, seg, ali, mnt = results[j]
            if np.sum(ali) == 0:
                have_alignment[j, 0, 0, 0] = 0
            image[j, :, :, 0] = img / 255.0
            segment[j, :, :, 0] = seg / 255.0
            alignment[j, :, :, 0] = ali / 255.0
            minutiae_w[j, (mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int), 0] = mnt[:, 0] % 8
            minutiae_h[j, (mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int), 0] = mnt[:, 1] % 8
            minutiae_o[j, (mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int), 0] = mnt[:, 2]
        # get seg
        label_seg = segment[:, ::8, ::8, :]
        label_seg[label_seg>0] = 1
        label_seg[label_seg<=0] = 0
        minutiae_seg = (minutiae_o!=-1).astype(float)
        # get ori & mnt
        orientation = tra_ori_model(alignment)        
        orientation = orientation/np.pi*180+90
        orientation[orientation>=180.0] = 0.0 # orientation [0, 180)
        minutiae_o = minutiae_o/np.pi*180+90 # [90, 450)
        minutiae_o[minutiae_o>360] = minutiae_o[minutiae_o>360]-360 # to current coordinate system [0, 360)
        minutiae_ori_o = np.copy(minutiae_o) # copy one
        minutiae_ori_o[minutiae_ori_o>=180] = minutiae_ori_o[minutiae_ori_o>=180]-180 # for strong ori label [0,180)      
        # ori 2 gaussian
        gaussian_pdf = signal.gaussian(361, 3)
        y = np.reshape(np.arange(1, 180, 2), [1,1,1,-1])
        delta = np.array(np.abs(orientation - y), dtype=int)
        delta = np.minimum(delta, 180-delta)+180
        label_ori = gaussian_pdf[delta]
        # ori_o 2 gaussian
        delta = np.array(np.abs(minutiae_ori_o - y), dtype=int)
        delta = np.minimum(delta, 180-delta)+180
        label_ori_o = gaussian_pdf[delta] 
        # mnt_o 2 gaussian
        y = np.reshape(np.arange(1, 360, 2), [1,1,1,-1])
        delta = np.array(np.abs(minutiae_o - y), dtype=int)  
        delta = np.minimum(delta, 360-delta)+180
        label_mnt_o = gaussian_pdf[delta]         
        # w 2 gaussian
        gaussian_pdf = signal.gaussian(17, 2)
        y = np.reshape(np.arange(0, 8), [1,1,1,-1])
        delta = (minutiae_w-y+8).astype(int)
        label_mnt_w = gaussian_pdf[delta]
        # h 2 gaussian
        delta = (minutiae_h-y+8).astype(int)
        label_mnt_h = gaussian_pdf[delta]
        # mnt cls label -1:neg, 0:no care, 1:pos
        label_mnt_s = np.copy(minutiae_seg)
        label_mnt_s[label_mnt_s==0] = -1 # neg to -1
        label_mnt_s = (label_mnt_s+ndimage.maximum_filter(label_mnt_s, size=(1,3,3,1)))/2 # around 3*3 pos -> 0
        # apply segmentation
        label_ori = label_ori * label_seg * have_alignment
        label_ori_o = label_ori_o * minutiae_seg
        label_mnt_o = label_mnt_o * minutiae_seg
        label_mnt_w = label_mnt_w * minutiae_seg
        label_mnt_h = label_mnt_h * minutiae_seg
        yield image, label_ori, label_ori_o, label_seg, label_mnt_w, label_mnt_h, label_mnt_o, label_mnt_s, batch_name
    if batch_size > 1 and use_multiprocessing==True:
        p.close()
        p.join()
    return

def get_maximum_img_size_and_names(dataset, sample_rate=None):
    if sample_rate is None:
        sample_rate = [1]*len(dataset)
    img_name, folder_name, img_size = [], [], []
    for folder, rate in zip(dataset, sample_rate):
        _, img_name_t = get_files_in_folder(folder+'images/', '.bmp')
        img_name.extend(img_name_t.tolist()*rate)
        folder_name.extend([folder]*img_name_t.shape[0]*rate)
        img_size.append(np.array(misc.imread(folder+'images/'+img_name_t[0]+'.bmp', mode='L').shape))
    img_name = np.asarray(img_name)
    folder_name = np.asarray(folder_name)
    img_size = np.max(np.asarray(img_size), axis=0)
    # let img_size % 8 == 0
    img_size = np.array(np.ceil(img_size/8)*8,dtype=np.int32)
    return img_name, folder_name, img_size

def sub_load_data(data, img_size, aug): 
    img_name, dataset = data
    img = misc.imread(dataset+'images/'+img_name+'.bmp', mode='L')
    seg = misc.imread(dataset+'seg_labels/'+img_name+'.png', mode='L')
    try:
        ali = misc.imread(dataset+'ori_labels/'+img_name+'.bmp', mode='L')
    except:
        ali = np.zeros_like(img)
    mnt = np.array(mnt_reader(dataset+'mnt_labels/'+img_name+'.mnt'), dtype=float)
    if any(img.shape != img_size):
        # random pad mean values to reach required shape
        if np.random.rand()<aug:
            tra = np.int32(np.random.rand(2)*(np.array(img_size)-np.array(img.shape)))
        else:
            tra = np.int32(0.5*(np.array(img_size)-np.array(img.shape)))
        img_t = np.ones(img_size)*np.mean(img)
        seg_t = np.zeros(img_size)
        ali_t = np.ones(img_size)*np.mean(ali)
        img_t[tra[0]:tra[0]+img.shape[0],tra[1]:tra[1]+img.shape[1]] = img
        seg_t[tra[0]:tra[0]+img.shape[0],tra[1]:tra[1]+img.shape[1]] = seg
        ali_t[tra[0]:tra[0]+img.shape[0],tra[1]:tra[1]+img.shape[1]] = ali
        img = img_t
        seg = seg_t
        ali = ali_t
        mnt = mnt+np.array([tra[1],tra[0],0]) 
    if np.random.rand()<aug:
        # random rotation [0 - 360] & translation img_size / 4
        rot = np.random.rand() * 360
        tra = (np.random.rand(2)-0.5) / 2 * img_size 
        img = ndimage.rotate(img, rot, reshape=False, mode='reflect')
        img = ndimage.shift(img, tra, mode='reflect')
        seg = ndimage.rotate(seg, rot, reshape=False, mode='constant')
        seg = ndimage.shift(seg, tra, mode='constant')
        ali = ndimage.rotate(ali, rot, reshape=False, mode='reflect')
        ali = ndimage.shift(ali, tra, mode='reflect') 
        mnt_r = point_rot(mnt[:, :2], rot/180*np.pi, img.shape, img.shape)  
        mnt = np.column_stack((mnt_r+tra[[1, 0]], mnt[:, 2]-rot/180*np.pi))
    # only keep mnt that stay in pic & not on border
    mnt = mnt[(8<=mnt[:,0])*(mnt[:,0]<img_size[1]-8)*(8<=mnt[:, 1])*(mnt[:,1]<img_size[0]-8), :]
    return img, seg, ali, mnt   

def merge_mul(x):
    return reduce(lambda x,y:x*y, x)
def merge_sum(x):
    return reduce(lambda x,y:x+y, x)
def reduce_sum(x):
    return K.sum(x,axis=-1,keepdims=True) 
def merge_concat(x):
    return torch.concat(x,3)
EPSILON = 1e-7
def select_max(x):
    x = x / (torch.sum(x,dim=-1,keepdim=True) + EPSILON)
    x = x.apply_(lambda x: (x) if (x>0.999) else (torch.zeros_like(x)))
    x = x / (torch.sum(x,dim=-1,keepdim=True) + EPSILON)
    return x  

def conv_bn_prelu(bottom, w_size, name, strides=(1,1), dilation_rate=(1,1)):
    if dilation_rate == (1,1):
        conv_type = 'conv'
    else:
        conv_type = 'atrousconv'
    #w = (64,3,3)
    top = torch.nn.functional.conv2d(bottom, \
        (w_size[0],w_size[0],w_size[1],w_size[2]), \
        stride = 1, padding = 'same', \
        dilation_rate = dilation_rate)
    top = nn.BatchNorm2d(top)
    top = nn.PReLU(top)
    return top

def get_main_net(input_shape=(512,512,1), weights_path=None):
    img_input=Input(input_shape)
    bn_img=Lambda(img_normalization, name='img_norm')(img_input)
    # feature extraction VGG
    class feat_extract_VGG(nn.Modules):
        def forward(self, input):
            conv = conv_bn_prelu(input,(64,3,3),'1_1')
            conv = conv_bn_prelu(conv,(64,3,3),'1_2')
            conv = nn.MaxPool2d(conv,kernel_size=(2,2),stride=2)
            
            conv = conv_bn_prelu(conv,(128,3,3),'2_1')
            conv = conv_bn_prelu(conv,(128,3,3),'2_2')
            conv = nn.MaxPool2d(conv,kernel_size=(2,2),stride=2)
            
            conv = conv_bn_prelu(conv,(256,3,3),'3_1')
            conv = conv_bn_prelu(conv,(256,3,3),'3_2')
            conv = conv_bn_prelu(conv,(256,3,3),'3_2')
            conv = nn.MaxPool2d(conv,kernel_size=(2,2),stride=2)
            return conv
        
    feat_extract_model = feat_extract_VGG()
    conv = feat_extract_model(bn_img)
    
    # multi-scale ASPP
    scale_1 = conv_bn_prelu(conv,(256,3,3), '4_1', dilation=(1,1))
    ori_1 = conv_bn_prelu(scale_1, (128,1,1), 'ori_1_1')
    ori_1 = nn.functional.conv2d(ori_1, (90,1,1,1), padding = 'same')#ori_1_2
    seg_1 = conv_bn_prelu(scale_1,(128,1,1),'seg_1_1')
    seg_1 = nn.functional.conv2d(seg_1, (1,1,1,1), padding='same')
    
    scale_2 = conv_bn_prelu(conv, (256,3,3), '4_2', dilation_rate=(4,4))
    ori_2 = conv_bn_prelu(scale_2, (128,1,1), 'ori_2_1')
    ori_2 = nn.functional.conv2d(seg_1,(90,1,1,1),padding='same')
    seg_2 = conv_bn_prelu(scale_2,(128,1,1),'seg_2_1')
    seg_2 = nn.functional.conv2d(seg_1,(1,1,1,1),padding='same')
    
    scale_3 = conv_bn_prelu(conv, (256,3,3), '4_3', dilation_rate=(8,8))
    ori_3 = conv_bn_prelu(scale_3, (128,1,1), 'ori_3_1')
    ori_3 = nn.functional.conv2d(ori_3,(90,1,1,1),padding='same')#ori_3_2
    seg_3=conv_bn_prelu(scale_3, (128,1,1), 'seg_3_1')
    seg_3 = nn.functional.conv2d(seg_3,(1,1,1,1),padding='same')#seg_3_2
    
    #sum fusion for ori
    class get_sum_fusion(nn.Module):
        def forward(self, input):
            return merge_sum(input)
    get_sum_model = get_sum_fusion()
    # sum fusion for ori
    ori_out = get_sum_model([ori_1,ori_2,ori_3])
    ori_out_1 = nn.functional.sigmoid(ori_out) #ori_out_1
    ori_out_2 = nn.functional.sigmoid(ori_out) #ori_out_2
    # sum fusion for segmentation
    seg_out = get_sum_model([seg_1,seg_2,seg_3])
    seg_out = nn.functional.sigmoid(seg_out) #seg_out
    # ----------------------------------------------------------------------------
    # enhance part
    filters_cos, filters_sin = gabor_bank(stride=2, Lambda=8)
    filters_cos = np.reshape(filters_cos,[90,1,25,25])
    filters_sin = np.reshape(filters_cos,[90,1,25,25])
    filter_img_real = nn.functional.conv2d(img_input,filters_cos, padding = 'same')#enh_img_real_1
    filter_img_imag = nn.functional.conv2d(img_input,filters_sin, padding = 'same')#enh_img_imag_1
        

    ori_peak = Lambda(ori_highest_peak)(ori_out_1)
    ori_peak = Lambda(select_max)(ori_peak) # select max ori and set it to 1
    upsample_ori = UpSampling2D(size=(8,8))(ori_peak)
    seg_round = Activation('softsign')(seg_out)      
    upsample_seg = UpSampling2D(size=(8,8))(seg_round)
    mul_mask_real = Lambda(merge_mul)([filter_img_real, upsample_ori])
    enh_img_real = Lambda(reduce_sum, name='enh_img_real_2')(mul_mask_real)
    mul_mask_imag = Lambda(merge_mul)([filter_img_imag, upsample_ori])
    enh_img_imag = Lambda(reduce_sum, name='enh_img_imag_2')(mul_mask_imag)
    enh_img = Lambda(atan2, name='phase_img')([enh_img_imag, enh_img_real])
    enh_seg_img = Lambda(merge_concat, name='phase_seg_img')([enh_img, upsample_seg])
    # ----------------------------------------------------------------------------
    # mnt part
    mnt_conv=conv_bn_prelu(enh_seg_img, (64,9,9), 'mnt_1_1') 
    mnt_conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(mnt_conv)

    mnt_conv=conv_bn_prelu(mnt_conv, (128,5,5), 'mnt_2_1') 
    mnt_conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(mnt_conv)

    mnt_conv=conv_bn_prelu(mnt_conv, (256,3,3), 'mnt_3_1')  
    mnt_conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(mnt_conv)    

    mnt_o_1=Lambda(merge_concat)([mnt_conv, ori_out_1])
    mnt_o_2=conv_bn_prelu(mnt_o_1, (256,1,1), 'mnt_o_1_1')
    mnt_o_3=Conv2D(180, (1,1), padding='same', name='mnt_o_1_2')(mnt_o_2)
    mnt_o_out=Activation('sigmoid', name='mnt_o_out')(mnt_o_3)

    mnt_w_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_w_1_1')
    mnt_w_2=Conv2D(8, (1,1), padding='same', name='mnt_w_1_2')(mnt_w_1)
    mnt_w_out=Activation('sigmoid', name='mnt_w_out')(mnt_w_2)

    mnt_h_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_h_1_1')
    mnt_h_2=Conv2D(8, (1,1), padding='same', name='mnt_h_1_2')(mnt_h_1)
    mnt_h_out=Activation('sigmoid', name='mnt_h_out')(mnt_h_2) 

    mnt_s_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_s_1_1')
    mnt_s_2=Conv2D(1, (1,1), padding='same', name='mnt_s_1_2')(mnt_s_1)
    mnt_s_out=Activation('sigmoid', name='mnt_s_out')(mnt_s_2)

    if args.mode == 'deploy':
        model = Model(inputs=[img_input,], outputs=[enh_img_real, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out])
    else:
        model = Model(inputs=[img_input,], outputs=[ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out])     
    if weights_path:
        model.load_weights(weights_path, by_name=True)
    return model

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
