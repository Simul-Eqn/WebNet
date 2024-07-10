path = './test1/' 
episode_count = 10000 
n_hiddens = 3000 
allowed_targets = list(range(9)) 
allowed_digits = list(range(10))
batch_size = 1 # 1 means unbatched 
# NOTE: batch_size>1 is currently unsupported 


import os 
try: 
    os.mkdir(path) 
except:
     pass 

import sys 
sys.path.insert(1, os.path.join(sys.path[0],'../')) 


import torch 
import torch.nn as nn 
from torchvision import transforms 

from mnist_utils import MultipleMNISTGenerator 
from extractors import EfficientNetExtractor 
from expanders import LinearsExpander 
from webnet import LSTMWebNet 

import random 
random.seed(10) 


imgsize = MultipleMNISTGenerator.default_final_size 
mnistgen = MultipleMNISTGenerator() 

n_inputs = 1000 
n_outputs = 1000 

extractor = EfficientNetExtractor(imgsize, n_inputs, os.path.join(path, 'extractor'), batched=(batch_size!=1)) 
lwn = LSTMWebNet(n_inputs, n_hiddens, n_outputs) 
expander = LinearsExpander(n_outputs, [4000, 16000], imgsize, nn.Sigmoid())

tfm = transforms.ToTensor() 
 

for episode in range(episode_count): 
    h = torch.zeros((n_hiddens)) 
    o = torch.zeros((n_outputs)) 
    h_c = lwn.h_cells_zeros() 
    o_c = lwn.o_cells_zeros() 

    losses = [] 

    start_delay = random.randint(10,20) 
    target_display = random.randint(3,6) 
    target_delay = random.randint(10,25) 
    options_display = random.randint(5,10) 

    target = random.choice(allowed_targets)
    rem = [i for i in allowed_digits if i != target] 
    options = [] 
    for i in range(random.randint(2,3)): 
        t = random.choice(rem) 
        rem.remove(t) 
        options.append(t) 


    empty_bceloss = nn.BCELoss(reduction='none') 
    for i in range(start_delay): 
        img = tfm(mnistgen.get_empty_image()) 
        #target = torch.zeros() 

        in_emb = extractor(img) 
        #print("IN EMB SHPE:", in_emb.shape)
        h, o, h_c, o_c = lwn(in_emb, h, o, h_c, o_c) 
        out = expander(o, to_img_shape=False) 

        loss = empty_bceloss(out, torch.zeros_like(out)) # since it's all supposed to be 0. 

        losses.append(loss) 
    
    # show target 
    for i in range(target_display): 
        img, bboxes = mnistgen.generate(target) 
        img = tfm(img) 
        #target = torch.zeros() 

        in_emb = extractor(img) 
        h, o, h_c, o_c = lwn(in_emb, h, o, h_c, o_c) 
        out = expander(o, to_img_shape=False) 

        loss = empty_bceloss(out, torch.zeros_like(out)) # since it's all supposed to be 0. 

        losses.append(loss) 

    # wait 
    for i in range(target_delay): 
        img = tfm(mnistgen.get_empty_image()) 
        #target = torch.zeros() 

        in_emb = extractor(img) 
        h, o, h_c, o_c = lwn(in_emb, h, o, h_c, o_c) 
        out = expander(o, to_img_shape=False) 

        loss = empty_bceloss(out, torch.zeros_like(out)) # since it's all supposed to be 0. 

        losses.append(loss) 


    # show choices 
    for i in range(target_display): 
        img, bboxes = mnistgen.generate(target, options) 
        img = tfm(img) 
        
        target = torch.zeros_like(img) 
        weights = torch.zeros_like(img) + 0.01 # don't care that much about mistakes outside 
        t_bbox = bboxes[0] 
        target[t_bbox[0]:t_bbox[2], t_bbox[1]:t_bbox[3]] = 1 
        weights[t_bbox[0]:t_bbox[2], t_bbox[1]:t_bbox[3]] = 1 
        for bbox in bboxes[1:]: 
            weights[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1 
        
        target = target.flatten().reshape(1,-1) # "fake" it as batched data to use weights 
        weights = weights.flatten() 

        bceloss = nn.BCELoss(weights, reduction='none') 

        in_emb = extractor(img) 
        h, o, h_c, o_c = lwn(in_emb, h, o, h_c, o_c) 
        out = expander(o, to_img_shape=False) 

        loss = bceloss(out.reshape(1,-1), target) 

        losses.append(loss) 




