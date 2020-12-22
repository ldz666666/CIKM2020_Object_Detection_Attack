import torch
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
import json

import cv2
import numpy as np
import os
from tool.darknet2pytorch import *
from infer import infer
from tqdm import tqdm
from skimage import measure
from collections import Counter
import pickle

from mmdet import __version__
from mmdet.apis import init_detector, inference_detector ,show_result_pyplot

pnum=1500

save_dir='./hard_seg_masks_all/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
img_dir='./select1000_new/'
files = os.listdir(img_dir)
files.sort()


def mask_iou(mask1,mask2):
    insection=np.logical_and(mask1,mask2)
    union=np.logical_or(mask1,mask2)
    return np.count_nonzero(insection)/np.count_nonzero(union)

def get_mask_list(mask_list,threshold=0.7):
    res=[]
    if len(mask_list)==0:
        print('empty mask_list')
        return 
    if len(mask_list)==1:
        return mask_list
    for m1 in mask_list:
        flag=0
        for m2 in res:
            if(mask_iou(m1,m2)>threshold):
                flag==1
                break
        if flag==0:
            res.append(m1)
    return res

def dilation(mask,x,y,patch_len):
    mask[int(max(0,y-0.5*patch_len)):int(min(mask.shape[0],y+0.5*patch_len)):2,int(max(0,x-0.5*patch_len)):int(min(mask.shape[1],x+0.5*patch_len)):2]=1.0
    mask[int(max(0,y-0.5*patch_len))+1:int(min(mask.shape[0],y+0.5*patch_len)):2,int(max(0,x-0.5*patch_len))+1:int(min(mask.shape[1],x+0.5*patch_len)):2]=1.0
    return mask

def deal_mask(mask,pixels):
    
    center_x,center_y=int(mask.shape[0]/2),int(mask.shape[1]/2)
    ones=np.zeros(mask.shape)
    patch_len=2
    pixels=min(np.count_nonzero(mask),pixels)
    
    print('pixels',pixels)
    
    while True:
    
        temp_ones=dilation(ones.copy(),center_x,center_y,patch_len)
        #print('temp_ones {} pixels'.format(np.count_nonzero(temp_ones)))
        union=np.logical_and(temp_ones,mask)
        #print('union {} pixels'.format(np.count_nonzero(union)))
        
        if np.count_nonzero(union)>pixels or patch_len >= mask.shape[0]:
            break;
        
        ones=temp_ones
        #print('ones {} pixels'.format(np.count_nonzero(ones)))
        
        patch_len+=2
    
    #print(np.count_nonzero(ones))
    result=np.logical_and(ones,mask)
    print('final {} pixels'.format(np.count_nonzero(result)))
    
    return result
    
def limit_patch(mask):
  
    label_mask=measure.label(mask,background=0,connectivity=2)
  
    ls=[]
    regs=measure.regionprops(label_mask)
    print('before , {} patches'.format(len(regs)))
    
    if len(regs)>10:
        for r in regs:
            ls.append([r.label,r.area])
        ls=np.array(ls)
        ls=ls[np.argsort(ls[:,1])[::-1],:]
        for i in range(10,ls.shape[0]):
             mask[label_mask==ls[i][0]]=0.0
    
    label_mask=measure.label(mask,background=0,connectivity=2)
    regs=measure.regionprops(label_mask)
    print('after , {} patches'.format(len(regs)))
    return mask         
    
        
    
   

    
if __name__=='__main__':   
    
    config = '/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
    checkpoint = './models/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
    rcnn_model=init_detector(config,checkpoint,device='cuda:0')
    
    save_dict={}
    
    for filename in files:
        
        print(filename)
        bboxes,bool_arr = inference_detector(rcnn_model, os.path.join(img_dir,filename))
        centers=[]
    
        masks=[]
        for mask in bool_arr:
            masks+=mask 
        masks=np.array(masks)
        bboxes=np.concatenate(bboxes)
        masks=masks[bboxes[:,4]>0.3,:]
        bboxes=bboxes[bboxes[:,4]>0.3,:]
        
        final_mask=np.zeros((500,500))
        masks=get_mask_list(masks)
        print(filename,len(masks),'masks')
        for m in masks:
            final_mask=np.logical_or(final_mask,deal_mask(m,int(pnum/len(masks))))
        #print(final_mask)
        
        '''
        img_name=os.path.join(save_dir,filename.split('.')[0])+'/before_mask.png'
        print('save',img_name,'before mask')
        cv2.imwrite(img_name,255*final_mask.astype(np.uint8))
        '''
        final_mask=limit_patch(final_mask)
        print('after limit')
        print(np.count_nonzero(final_mask),'pixels')
        label_mask=measure.label(final_mask,background=0,connectivity=2)
        regs=measure.regionprops(label_mask)
        print(len(regs),'patches')
        
        if not os.path.exists(os.path.join(save_dir,filename.split('.')[0])):
            os.makedirs(os.path.join(save_dir,filename.split('.')[0]))
        img_name=os.path.join(save_dir,filename.split('.')[0])+'/dilation_mask.png'
        print('save',img_name,'mask')
        cv2.imwrite(img_name,255*final_mask.astype(np.uint8))
        
        save_dict[filename]=final_mask
    
    pickle.dump(save_dict,open('dilation_mask_{}.pkl'.format(pnum),'wb'))
    print('dict saved')

     
