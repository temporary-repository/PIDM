import pdb
import numpy as np 
import cv2 
import torch 
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.init as init
import torch.nn as nn
from pytorch_ssim.pytorch_ssim import SSIM
ssim = SSIM()
import random
import os 

def seed_torch(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def xavier_init(layer):
    if isinstance(layer, nn.Conv2d):
        init.xavier_uniform_(layer.weight)  # 使用Xavier初始化权重

def he_init(layer):
    if isinstance(layer, nn.Conv2d):
        init.kaiming_uniform_(layer.weight)

def vis_PSF(psf, save_name=''):
    plt.figure(figsize=(7, 7))
    tmp = sns.heatmap(psf, cmap='turbo', vmax=None, annot=False, xticklabels=False, 
        yticklabels=False, cbar=False, linewidths=0.0, rasterized=True)
    tmp.figure.savefig(save_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    # tmp.figure.savefig(save_name, bbox_inches='tight', pad_inches=0.0)
    plt.close()

# save false-color img
def gen_false_color_img(img, save_name='', clist=[30, 60, 90]):
    h, w, band = img.shape
    img_rgb = np.zeros((h, w, 3))
    img = np.array(img, dtype=np.float64)
  
    for b in range(band):
        if b not in clist: continue
        img_b = img[..., b]
        img255 = img_b * 255
        # false color
        if b == clist[0]:
            img_rgb[:, :, 0] = img255[:]
        if b == clist[1]:
            img_rgb[:, :, 1] = img255[:]
        if b == clist[2]:
            img_rgb[:, :, 2] = img255[:]
   
        img_rgb = cv2.cvtColor(np.uint8(img_rgb), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(7, 7))  
        plt.axis('off')  
        plt.imshow(img_rgb)
        plt.savefig(save_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
        plt.close()