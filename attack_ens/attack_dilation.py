# Helper function for extracting features from pre-trained models
import sys, os
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
import pickle

from method_dilation import Attacker
from loader_new import Coco
from tool.darknet2pytorch import Darknet
from tool.torch_utils import do_detect

from mmdet.apis import init_detector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/attack_ens/output_212_400_1b3/', type=str, help='path to data')
    parser.add_argument('--output_dir', default='./output_1b3_400/', type=str, help='path to results')
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')
    parser.add_argument('--steps', default=400, type=int, help='iteration steps')
    parser.add_argument('--thres', default=0.0, type=float, help='conf thres in loss function')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    #model
    cfgfile = "models/yolov4.cfg"
    weightfile = "models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    
    config = '/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    rcnn_model = init_detector(config, checkpoint, device=torch.device('cuda'))
    
    # set dataset
    dataset = Coco(args.input_dir)
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=args.batch_size, 
                                         shuffle=False)

    # set attacker
    attacker = Attacker(steps=args.steps, thres=args.thres, device=torch.device('cuda'))
    #num = 0

    for ind, (img, filenames) in enumerate(loader):

        # run attack
        #adv = attacker.attack(darknet_model, img.cuda())
        #print('Attack on {}:'.format(os.path.split(filenames[0])[-1]))
        patch = attacker.attack(darknet_model, rcnn_model, filenames, img.cuda())
        
        #output = do_detect(darknet_model, img.cuda() + patch, 0.5, 0.4, True)
        #print('detected:{}'.format(output))
        #num = num + len(output[0]) + len(output[1]) + len(output[2]) + len(output[3])
        
        # save results
        for bind, filename in enumerate(filenames):
            out_img = patch[bind].detach().cpu().numpy()
            print('Attack on {}:'.format(os.path.split(filename)[-1]))
            
            
            out_img = np.transpose(out_img, axes=[1, 2, 0]) * 255.0
            out_img = out_img[:, :, ::-1]
            
            #size = int(500.0/608.0*out_img.shape[0])
            #size = 500
            #sized = cv2.resize(out_img, (size, size))
            
            
            img_path = os.path.join(args.input_dir, filename)
            img = cv2.imread(img_path)
            #print(img.shape)
            adv = img + out_img
            
            #img[int(0.5*(img.shape[0]-sized.shape[0])):int(0.5*(img.shape[0]-sized.shape[0]))+sized.shape[0], int(0.5*(img.shape[1]-sized.shape[1])):int(0.5*(img.shape[1]-sized.shape[1]))+sized.shape[1],:] += sized.astype('uint8')
            out_filename = os.path.join(output_dir, os.path.split(filename)[-1])
            cv2.imwrite(out_filename, adv)
    #print(num)