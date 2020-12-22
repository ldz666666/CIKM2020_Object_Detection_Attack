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
from infer import infer,infer_rcnn
from tqdm import tqdm
from skimage import measure
from collections import Counter
import pickle
from skimage import measure

from mmdet import __version__
from mmdet.apis import init_detector, inference_detector ,show_result_pyplot
from eval import connected_domin_detect_and_score, count_connected_domin_score_single,infer_yolo



clean_dir='./select1000_new/'
patch_dir='/eval_code/8_25_random_expand/'

files=os.listdir(clean_dir)
files.sort()

tmp_ori_dir='./tmp_ori_dir/'
tmp_patch_dir='./8_27_expand_shrink_1/'

if not os.path.exists(tmp_ori_dir):
    os.makedirs(tmp_ori_dir)
if not os.path.exists(tmp_patch_dir):
    os.makedirs(tmp_patch_dir)


def cal_score(img_0,img_1,yolo_model,rcnn_model):
    
    domain_score=count_connected_domin_score_single(img_0,img_1)    
    yolo_score=infer_yolo(img_0,img_1,yolo_model)
    rcnn_score=infer_rcnn(img_0,img_1,rcnn_model)
    return domain_score*(yolo_score+rcnn_score)

        
def shrink_pinjie(img_ori,img_patch,step,size):
    img_res=img_ori.copy()
    img_res[step:size-step , step:size-step ,:]=img_patch[step:size-step,step:size-step,:]
    return img_res


def shrink_optim_exp(filename,yolo_model,rcnn_model,cln_dir,atk_dir,out_dir):
    
    print()
    print('optimizing',filename)
    
    img_ori=cv2.imread(os.path.join(clean_dir,filename))
    img_patch=cv2.imread(os.path.join(atk_dir,filename))
    
    #img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    #img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
    
    #after that , img_ori ,img_patch are numpy arrays after rgb converting
    
    img_res=img_patch.copy()
    size=img_ori.shape[1]
    
    #calculate img_patch score
    
    print('calculating ori patch score')
    score_ori=cal_score(os.path.join(cln_dir,filename),os.path.join(atk_dir,filename),yolo_model,rcnn_model)
    
    print('origin score',score_ori)
    if score_ori==0:
        return 0
    
    
    score_res=score_ori

    for step in range(0,int(size/2)-1):
        
        #connect image
        img_tmp=shrink_pinjie(img_ori,img_patch,step,size)
        if (img_tmp==img_patch).all():
            #print('step {} unecessary'.format(step))
            continue
        
        '''
        label_tmp=measure.label(img_tmp, background=0, connectivity=2)
        reg=measure.regionprops(label_tmp)
        if len(reg)==0:
            print('dont have patch')
            break
        '''
        #write
        cv2.imwrite(os.path.join(out_dir,filename),img_tmp)
        #calculate score
        score_tmp=cal_score(os.path.join(cln_dir,filename),os.path.join(out_dir,filename),yolo_model,rcnn_model)
        print('score_tmp',score_tmp)
        
        if score_tmp>score_res:
            score_res=score_tmp
            img_res=img_tmp
            
        if score_tmp<score_ori:
            break
    
    print('after optim score',score_res)
    print('delta',score_res-score_ori)
    print()
    cv2.imwrite(os.path.join(out_dir,filename),img_res)
    return score_res-score_ori




def shrink_optim(filename,yolo_model,rcnn_model):
    
    print()
    print('optimizing',filename)
    
    img_ori=cv2.imread(os.path.join(clean_dir,filename))
    img_patch=cv2.imread(os.path.join(patch_dir,filename))
    
    #img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    #img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
    
    #after that , img_ori ,img_patch are numpy arrays after rgb converting
    
    img_res=img_patch.copy()
    size=img_ori.shape[1]
    
    #calculate img_patch score
    
    print('calculating ori patch score')
    score_ori=cal_score(os.path.join(clean_dir,filename),os.path.join(patch_dir,filename),yolo_model,rcnn_model)
    
    print('origin score',score_ori)
    
    score_res=score_ori

    for step in range(0,int(size/2)-1):
        
        #connect image
        img_tmp=shrink_pinjie(img_ori,img_patch,step,size)
        if (img_tmp==img_patch).all():
            #print('step {} unecessary'.format(step))
            continue
        
        '''
        label_tmp=measure.label(img_tmp, background=0, connectivity=2)
        reg=measure.regionprops(label_tmp)
        if len(reg)==0:
            print('dont have patch')
            break
        '''
        #write
        cv2.imwrite(os.path.join(tmp_patch_dir,filename),img_tmp)
        #calculate score
        score_tmp=cal_score(os.path.join(clean_dir,filename),os.path.join(tmp_patch_dir,filename),yolo_model,rcnn_model)
        print('score_tmp',score_tmp)
        
        if score_tmp>score_res:
            score_res=score_tmp
            img_res=img_tmp
            
        if score_tmp<score_ori:
            break
    
    print('after optim score',score_res)
    print('delta',score_res-score_ori)
    print()
    cv2.imwrite(os.path.join(tmp_patch_dir,filename),img_res)
    return score_res-score_ori

    
if __name__=='__main__':   
    
    cfgfile = "models/yolov4.cfg"
    weightfile = "models/yolov4.weights"
    yolo_model = Darknet(cfgfile)
    yolo_model.load_weights(weightfile)
    yolo_model = yolo_model.eval().cuda()
    
    config = '/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    rcnn_model = init_detector(config, checkpoint, device=torch.device('cuda'))
    
    print('model loaded')
    
    delta=0
    for filename in files:
        delta+=shrink_optim(filename,yolo_model,rcnn_model)
        
    print(delta)
        
