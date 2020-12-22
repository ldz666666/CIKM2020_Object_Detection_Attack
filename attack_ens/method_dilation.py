from typing import Optional, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

import sys
import pickle

from tool.torch_utils import do_detect
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

import math

class Resize(nn.Module):
    def __init__(self, height, width):
        super(Resize, self).__init__()
        self.height = height
        self.width = width
        self.resize_layer = nn.Upsample(size=(self.height, self.width), mode='bilinear').cuda()

    def forward(self, x):
        resized = self.resize_layer(x)
        return resized
        
class Normalize(nn.Module):

    def __init__(self):
        super(Normalize, self).__init__()

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1))

    def forward(self, x):
        return (x - autograd.Variable(self.mean).cuda()) / autograd.Variable(self.std).cuda()
                
class Attacker:
    def __init__(self,
                 steps: int,
                 thres: float,
                 device: torch.device = torch.device('cpu')) -> None:
        self.steps = steps
        self.device = device
        self.thres = thres
        
    def attack(self, 
               model_yolo: nn.Module, 
               model_rcnn: nn.Module,
               filenames,inputs: torch.Tensor)-> torch.Tensor:

        #load rcnn boxes
        box_dict=pickle.load(open('dilation_mask.pkl', 'rb'))
        
        batch_size = inputs.shape[0]
        
        #perturbation_size = int(0.14 * inputs.size(3))
        #delta = torch.zeros([inputs.size(0), inputs.size(1), perturbation_size, perturbation_size], requires_grad=True, device=torch.device('cuda'))
        
        delta = torch.zeros_like(inputs, requires_grad=True)
        #delta1 = torch.zeros_like(inputs, requires_grad=True)
        #delta2 = torch.zeros_like(inputs, requires_grad=True)
        
        #resizer
        resizer_yolo = Resize(608, 608)
        resizer_rcnn = Resize(800, 800)
        normalizer = Normalize()
        
        #different kind of mask
        mask = torch.zeros_like(inputs)
        #mask1 = torch.zeros_like(inputs)
        #mask2 = torch.zeros_like(inputs)
        #perturb = inputs.clone()
        #length = int(0.14 * inputs.size(3))
        
        #mask[:,:,int(0.5*(mask.size(2)-length)):int(0.5*(mask.size(2)-length))+length,int(0.5*(mask.size(3)-length)):int(0.5*(mask.size(3)-length))+length] = 1.0
        #mask1[:,:,int(0.5*(mask.size(2)-length)):int(0.5*(mask.size(2)-length))+length,int(0.5*(mask.size(3)-length)):int(0.5*mask.size(3))] = 1.0
        #mask2[:,:,int(0.5*(mask.size(2)-length)):int(0.5*(mask.size(2)-length))+length,int(0.5*mask.size(3))+1:int(0.5*(mask.size(3)-length))+length] = 1.0
        
        
        '''output_refer=do_detect(model_yolo, resizer_yolo(inputs), 0.5, 0.4, True)
        for ind in range(len(output_refer)):
            bbox=np.array(output_refer[ind])
            num_obj=bbox.shape[0]
            patch_len=int(math.sqrt(5000.0/num_obj))
            for i in range(num_obj):
                x,y=500*bbox[i][0],500*bbox[i][1]
                mask[ind,:,int(max(0,y-0.5*patch_len)):int(min(mask.size(2),y+0.5*patch_len)),int(max(0,x-0.5*patch_len)):int(min(mask.size(3),x+0.5*patch_len))]=1.0'''
        #use rcnn masks
        for ind,fname in enumerate(filenames):
            mask[ind]=torch.tensor(box_dict[fname]).cuda()
                
        '''for ind,fname in enumerate(filenames):
            result_p=box_dict[fname]
            result_p=result_p[:min(result_p.shape[0],10),:]
            num_obj=result_p.shape[0]
            patch_len=int(math.sqrt(5000.0/max(1,num_obj)))
            if result_p.shape[0] !=0:
                for j in range(result_p.shape[0]):
                    x1,y1,x2,y2=result_p[j][0],result_p[j][1],result_p[j][2],result_p[j][3]
                    x,y=int(0.5*(x1+x2)),int(0.5*(y1+y2))
                    mask[ind,:,int(max(0,y-0.5*patch_len)):int(min(mask.size(2),y+0.5*patch_len)),int(max(0,x-0.5*patch_len)):int(min(mask.size(3),x+0.5*patch_len))]=1.0
            else:
                print('img {} find 0 objs,put patch in the middle'.format(fname))
                x,y=250,250
                patch_len=70
                mask[ind,:,int(max(0,y-0.5*patch_len)):int(min(mask.size(2),y+0.5*patch_len)),int(max(0,x-0.5*patch_len)):int(min(mask.size(3),x+0.5*patch_len))]=1.0'''
                
        
        
        # setup optimizer
        optimizer = optim.SGD([delta], lr=0.1, momentum=0.9)
        #optimizer1 = optim.SGD([delta1], lr=1, momentum=0.9)
        #optimizer2 = optim.SGD([delta2], lr=1, momentum=0.9)

        # for choosing best results
        #best_loss = 1e4 * torch.ones(inputs.size(0), dtype=torch.float, device=torch.device('cuda'))
        #best_delta = torch.zeros_like(delta)
        
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        target_yolo = torch.zeros([inputs.size(0), 22743], device=torch.device('cuda'))
        target_rcnn = torch.zeros([inputs.size(0), 1000], device=torch.device('cuda'))

        for i in range(self.steps):
            adv = inputs + delta * mask
            if i%2==0:
                #adv = perturb + delta * mask1
                adv_yolo = resizer_yolo(adv)
                output_yolo = model_yolo(adv_yolo)
                confs_yolo = output_yolo[:, :, 4:]
                #max_conf_yolo, _ = torch.max(confs_yolo, axis=2) #batch, num
                max_conf_yolo = torch.sum(confs_yolo, axis=2) #batch, num
                loss = criterion(max_conf_yolo, target_yolo)
                #perturb += delta.data * mask1
            else:
                #adv = perturb + delta * mask2
                adv_rcnn = resizer_rcnn(adv)
                adv_rcnn = normalizer(adv_rcnn)
                #print(adv_rcnn.size())
                data = {'img_metas': [[{'filename': 'test.png', 'ori_filename': 'test.png', 'ori_shape': (500, 500, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1.6, 1.6, 1.6, 1.6], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([0.0, 0.0, 0.0], dtype=np.float32), 'std': np.array([1.0, 1.0, 1.0], dtype=np.float32), 'to_rgb': True}}]],'img': []}
                data['img'].append(adv_rcnn)
                
 
                output_rcnn = model_rcnn(return_loss=False, rescale=False, **data)
                #print(output_rcnn.size())
                confs_rcnn = output_rcnn[:, :80]
                #print(confs_rcnn.shape)
                #max_conf_rcnn, _ = torch.sum(confs_rcnn, axis=1) #batch, num
                max_conf_rcnn = torch.sum(confs_rcnn, axis=1) #batch, num
                #print(max_conf_rcnn)
                loss = criterion(max_conf_rcnn.unsqueeze(0), target_rcnn)
                #perturb += delta.data * mask2

            loss = torch.mean(loss, axis=1)
            #print(loss)
            
            '''is_better = loss < best_loss

            best_loss[is_better] = loss[is_better]
            best_delta[is_better] = delta.data[is_better]
            #print(best_loss)'''
            
            loss = torch.mean(loss)
            optimizer.zero_grad()
            #optimizer1.zero_grad()
            #optimizer2.zero_grad()
            loss.backward()
            
            #print(delta)
            '''if i%2==0:
                delta.grad.data = delta.grad.data * mask1
            else:
                delta.grad.data = delta.grad.data * mask2'''

            # renorm gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=float('inf'), dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                
                

            optimizer.step()
            
            
            # avoid out of bound
            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)
            

        return delta * mask
        
