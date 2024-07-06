path = './test1/' 
episode_count = 10000 
n_hiddens = 2000 


import os 
try: 
    os.mkdir(path) 
except:
     pass 


import torch 
from torchvision import transforms 

from mnist_test.mnist_utils import MultipleMNISTGenerator 
from mnist_test.extractors import EfficientNetExtractor 
from mnist_test.expanders import LinearsExpander 
from webnet import LSTMWebNet 

import random 
random.seed(10) 


imgsize = MultipleMNISTGenerator.default_final_size 

extractor = EfficientNetExtractor(imgsize, 1000, os.path.join(path, 'extractor')) 
lwn = LSTMWebNet(1000, n_hiddens, 1000) 
expander = LinearsExpander(1000, [4000, 16000], imgsize)

tfm = transforms.ToTensor() 


for episode in range(episode_count): 
    start_delay = random.randint(10,20) 
    h = torch.zeros((n_hiddens, 1)) 
    o = torch.zeros((n_hiddens, 1)) 
    h_c = lwn.h_cells_zeros() 
    o_c = lwn.o_cells_zeros() 

    for i in range(start_delay): 
        img = tfm(MultipleMNISTGenerator.get_empty_image()) 
        target = torch.zeros() 

        in_emb = extractor() 
        h, o, h_c, o_c = lwn(in_emb) 
        out = expander(o) 






