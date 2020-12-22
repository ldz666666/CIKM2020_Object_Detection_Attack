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
from shrink import shrink_optim_exp

#tmp_ori_dir='./tmp_ori_dir/'
out_dir='./output_3000_pix_es/'
cln_dir='./select1000_new/'
atk_dir='/attack_ens/output_3000_pix_200iter/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)




files_atk=os.listdir(atk_dir)
files_out=os.listdir(out_dir)
files_final=[f for f in files_atk if f not in files_out]
files_final.sort()
print(files_final)

def cal_score(img_0,img_1,yolo_model,rcnn_model):
    
    domain_score=count_connected_domin_score_single(img_0,img_1)    
    yolo_score=infer_yolo(img_0,img_1,yolo_model)
    rcnn_score=infer_rcnn(img_0,img_1,rcnn_model)
    return domain_score*(yolo_score+rcnn_score)

        
def pinjie(img_ori,img_patch,step,size):
    #img_res=img_ori.copy()
    img_res=img_patch.copy()
    #change: shrink: outward is clean , expand: inward is clean
    #img_res[step:size-step , step:size-step ,:]=img_patch[step:size-step,step:size-step,:]
    img_res[step:size-step , step:size-step ,:]=img_ori[step:size-step,step:size-step,:]
    return img_res

def pinjie_center(img_ori,img_patch,center_x,center_y,step):
    
    size=img_ori.shape[1]
    img_res=img_patch.copy()
    img_res[max(center_x-step,0):min(center_x+step,size), max(center_y-step,0):min(center_y+step,size),:]=img_ori[max(center_x-step,0):min(center_x+step,size), max(center_y-step,0):min(center_y+step,size),:]
    return img_res


def iter_img(cord,filename,score_ori,score_res,img_res,img_ori,img_patch,yolo_model,rcnn_model):
    #for step in range(0,int(size/2)-1):
    size=img_ori.shape[1]
    
    for step in range(0,int(size/2)-1):
    
        #connect image
        img_tmp=pinjie_center(img_ori,img_patch,cord[0],cord[1],step)
        if (img_tmp==img_patch).all():
            #print('step {} unecessary'.format(step))
            continue
        
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
            
    return img_res , score_res


def expand_optim(filename,yolo_model,rcnn_model):
    
    print()
    print('optimizing',filename)
    
    img_ori=cv2.imread(os.path.join(cln_dir,filename))
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
    
    score_res=score_ori
    if score_ori==0:
        return 0

    delta_img=img_patch-img_ori
    label_delta=measure.label(delta_img,connectivity=2)
    reg=measure.regionprops(label_delta)
    temp_cord=[0,0]
    for r in reg:
        temp_cord[0]+=r.centroid[0]
        temp_cord[1]+=r.centroid[1]
    temp_cord[0]=int(temp_cord[0]/len(reg))
    temp_cord[1]=int(temp_cord[1]/len(reg))
    
    print('zhixin')
    img_res , score_res=iter_img(temp_cord,filename,score_ori,score_res,img_res,img_ori,img_res,yolo_model,rcnn_model)
    
    temp_cord=(int(size/2),int(size/2))
    print('center')
    img_res , score_res=iter_img(temp_cord,filename,score_res,score_res,img_res,img_ori,img_res,yolo_model,rcnn_model)
    
    temp_cord=(0,0)
    print('zero')
    img_res , score_res=iter_img(temp_cord,filename,score_res,score_res,img_res,img_ori,img_res,yolo_model,rcnn_model)
    
    for i in range(3):
        print('random {}'.format(i+1))
        temp_cord=list((size*np.random.rand(2)).astype(np.uint8))
        print(temp_cord)
        img_res , score_res=iter_img(temp_cord,filename,score_res,score_res,img_res,img_ori,img_res,yolo_model,rcnn_model)
    
    print('after optim score',score_res)
    print('delta is',score_res-score_ori)
    print()
    cv2.imwrite(os.path.join(out_dir,filename),img_res)
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
    for filename in files_final:
        delta+=expand_optim(filename,yolo_model,rcnn_model)
    for filename in files_final:
        delta+=shrink_optim_exp(filename,yolo_model,rcnn_model,cln_dir,atk_dir,out_dir)
    print('improved score is',delta)
