import os 

import torch 
import torch.nn as nn 
from torchvision import transforms 
torch.manual_seed(10) 
torch.cuda.manual_seed(10) 

from efficientnet_pytorch.model import EfficientNet 
from efficientnet_pytorch.utils import round_filters 

from mnist_utils import MultipleMNISTGenerator 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class EfficientNetExtractor(nn.Module): 
    default_in_size = MultipleMNISTGenerator.default_final_size[0] * MultipleMNISTGenerator.default_final_size[1] 
    def __init__(self, in_size=None, out_size=1000, save_dir='./enet_extractor/', batched:bool=True, device=device): 

        nn.Module.__init__(self) 
        
        if in_size is None: in_size = EfficientNetExtractor.default_in_size 
        self.in_size = in_size 
        self.out_size = out_size 
        self.save_dir = save_dir 
        self.batched = batched 

        self.efficientnet = EfficientNet.from_name('efficientnet-b2', in_channels=1).to(device) # out: 1000 
        self.dense = nn.Linear(in_features=16000, out_features=out_size, device=device) 

        if not batched: 
            self.efficientnet._bn0 = nn.InstanceNorm2d(num_features = round_filters(32, self.efficientnet._global_params), 
                                                       momentum = 1 - self.efficientnet._global_params.batch_norm_momentum, 
                                                       eps = self.efficientnet._global_params.batch_norm_epsilon) 
            
            self.efficientnet._bn1 = nn.InstanceNorm2d(num_features = round_filters(1280, self.efficientnet._global_params), 
                                                       momentum = 1 - self.efficientnet._global_params.batch_norm_momentum, 
                                                       eps = self.efficientnet._global_params.batch_norm_epsilon) 
            
            for block in self.efficientnet._blocks: 
                oup = block._block_args.input_filters * block._block_args.expand_ratio 
                if block._block_args.expand_ratio != 1: 
                    block._bn0 = nn.InstanceNorm2d(num_features = oup, momentum = block._bn_mom, eps = block._bn_eps) 
                block._bn1 = nn.InstanceNorm2d(num_features = oup, momentum = block._bn_mom, eps = block._bn_eps)
                block._bn2 = nn.InstanceNorm2d(num_features = block._block_args.output_filters, momentum = block._bn_mom, eps = block._bn_eps) 


            # testing 
            #print("BATCHED:", batched) 
            #print(self.efficientnet._bn0) 
            #print(self.efficientnet._bn1) 
            #for block in self.efficientnet._blocks: 
            #    print(block) 
                


    def forward(self, img): 
        return self.dense(self.efficientnet(img).flatten()) 
    
    def save(self, save_dir=None): 
        if save_dir is None: save_dir = self.save_dir 
        torch.save(self.efficientnet.state_dict(), os.path.join(save_dir, 'enet_model.bin')) 
        torch.save(self.dense, os.path.join(save_dir, 'extractor_layer.bin'))

    def load(self, dir=None): 
        if dir is None: dir = self.save_dir 
        self.efficientnet.load_state_dict(torch.load(os.path.join(dir, 'enet_model.bin'))) 
        self.dense.load_state_dict(torch.load(os.path.join(dir, 'extractor_layer.bin')))
        

         


