import os 

import torch 
import torch.nn as nn 
from torchvision import transforms 
torch.manual_seed(100) 
torch.cuda.manual_seed(100) 

from efficientnet_pytorch import EfficientNet 

from mnist_utils import MultipleMNISTGenerator 


class EfficientNetExtractor(nn.Module): 
    default_in_size = MultipleMNISTGenerator.default_final_size[0] * MultipleMNISTGenerator.default_final_size[1] 
    def __init__(self, in_size=None, out_size=1000, save_dir='./enet_extractor/'): 
        
        if in_size is None: in_size = EfficientNetExtractor.default_in_size 
        self.in_size = in_size 
        self.out_size = out_size 
        self.save_dir = save_dir 

        self.efficientnet = EfficientNet.from_name('efficientnet-b2', in_channels=1) # out: 1000 
        self.dense = nn.Linear(in_features=1000, out_features=out_size) 
    
    def forward(self, img): 
        return self.dense(self.efficientnet(img)) 
    
    def save(self, save_dir=None): 
        if save_dir is None: save_dir = self.save_dir 
        torch.save(self.efficientnet.state_dict(), os.path.join(save_dir, 'enet_model.bin')) 
        torch.save(self.dense, os.path.join(save_dir, 'extractor_layer.bin'))

    def load(self, dir=None): 
        if dir is None: dir = self.save_dir 
        self.efficientnet.load_state_dict(torch.load(os.path.join(dir, 'enet_model.bin'))) 
        self.dense.load_state_dict(torch.load(os.path.join(dir, 'extractor_layer.bin')))
        

         


