import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class Coco(Dataset):
    def __init__(self, root_dir):
        labels_dir = os.path.join(root_dir)
        self.image_dir = root_dir
        self.files = os.listdir(root_dir)
        self.files.sort()

    def __len__(self):
        l = len(self.files)
        return l

    def __getitem__(self, idx):
        #pil
        '''img_name = self.files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        img = Image.open(img_path).convert('RGB')
        
        resize_small = transforms.Compose([transforms.Resize((608, 608)),transforms.ToTensor()])
        img_resize = resize_small(img)

        return img_resize, img_name'''
        
        #cv2
        img_name = self.files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_path)
        #sized = cv2.resize(img, (608, 608))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        out_img = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().div(255.0)
        
        return out_img, img_name
        