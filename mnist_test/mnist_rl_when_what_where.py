path = './test1/' 
episode_count = 10000 
n_hiddens = 5000 
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
torch.manual_seed(10) 
torch.cuda.manual_seed(10) 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 


imgsize = MultipleMNISTGenerator.default_final_size 
mnistgen = MultipleMNISTGenerator() 

n_inputs = 1000 
n_outputs = 1000 

extractor = EfficientNetExtractor(imgsize, n_inputs, os.path.join(path, 'extractor'), batched=(batch_size!=1), device=device) 
lwn = LSTMWebNet(n_inputs, n_hiddens, n_outputs, device=device).to(device) 
expander = LinearsExpander(n_outputs, [10000], imgsize, nn.Sigmoid()).to(device) 

total_params = sum(p.numel() for p in extractor.parameters()) + sum(p.numel() for p in lwn.parameters()) + sum(p.numel() for p in expander.parameters())


tfm = transforms.ToTensor() # TODO: perhaps more augmentation? 


optimizer = torch.optim.Adam([
    {'params': extractor.parameters(), 'lr': 1e-8}, 
    {'params': lwn.get_non_lstm_params(), 'lr': 1e-8}, 
    {'params': lwn.get_lstm_params(), 'lr': 2e-13}, 
    {'params': expander.parameters(), 'lr': 1e-8}, 
])

 

for episode in range(episode_count): 
    h = torch.zeros((n_hiddens)).to(device) 
    o = torch.zeros((n_outputs)).to(device) 
    h_c = lwn.h_cells_zeros(device=device)
    o_c = lwn.o_cells_zeros(device=device)

    optimizer.zero_grad() 

    #losses = [] 

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


    empty_bceloss = nn.BCELoss(reduction='mean') 

    
    ''' 
    proof of concept 
    import numpy as np 
    print("APPROACHING 1:")
    for i in np.linspace(0.9, 1.0, 20, endpoint=False): print(empty_bceloss(torch.tensor([i], dtype=torch.float32), torch.tensor([1.0])))
    print() 
    print("APPROACHING 0:")
    for i in np.linspace(0.1, 0.0, 20, endpoint=False): print(empty_bceloss(torch.tensor([i], dtype=torch.float32), torch.tensor([0.0])))
    print() 
    print() 
    as long as labels are either 0.0 or 1.0, loss can be basically 0. 
    ''' 

    losses = [] # keep track of scalar losses to report average loss 

    optimizer.zero_grad() 
    torch.cuda.empty_cache()

    # single episode 
    for i in range(start_delay): 
        img = tfm(mnistgen.get_empty_image()).to(device) 

        in_emb = extractor(img) 
        #print("IN EMB SHPE:", in_emb.shape)
        h, o, h_c, o_c = lwn(in_emb, h, o, h_c, o_c) 
        out = expander(o, to_img_shape=False) 

        loss = empty_bceloss(out, torch.zeros_like(out))  # since it's all supposed to be 0. 
        loss.backward(retain_graph=True) # not updating weights yet, just adding to gradients 
        
        losses.append(loss.item()) 
        del loss 

        torch.cuda.empty_cache()

        #print(torch.cuda.memory_allocated()) 

    # show target 
    for i in range(target_display): 
        img, bboxes = mnistgen.generate(target) 
        img = tfm(img).to(device) 
        #target = torch.zeros() 

        in_emb = extractor(img) 
        h, o, h_c, o_c = lwn(in_emb, h, o, h_c, o_c) 
        out = expander(o, to_img_shape=False) 

        loss = empty_bceloss(out, torch.zeros_like(out))  # since it's all supposed to be 0. 
        loss.backward(retain_graph=True) # not updating weights yet, just adding to gradients 
        
        losses.append(loss.item())
        del loss 

        torch.cuda.empty_cache()

    # wait 
    for i in range(target_delay): 
        img = tfm(mnistgen.get_empty_image()).to(device) 
        #target = torch.zeros() 

        in_emb = extractor(img) 
        h, o, h_c, o_c = lwn(in_emb, h, o, h_c, o_c) 
        out = expander(o, to_img_shape=False) 

        loss = empty_bceloss(out, torch.zeros_like(out))  # since it's all supposed to be 0. 
        loss.backward(retain_graph=True) # not updating weights yet, just adding to gradients 
        
        losses.append(loss.item())
        del loss 
        
        torch.cuda.empty_cache()


    # show choices 
    bceloss = nn.BCELoss(reduction='none') 
    for i in range(target_display): 
        img, bboxes = mnistgen.generate(target, options) 
        img = tfm(img).to(device) 
        
        train_target = torch.zeros_like(img) 
        weights = torch.zeros_like(img, requires_grad=False) + 0.01 # don't care that much about mistakes outside 
        t_bbox = bboxes[0] 
        train_target[t_bbox[0]:t_bbox[2], t_bbox[1]:t_bbox[3]] = 1 
        weights[t_bbox[0]:t_bbox[2], t_bbox[1]:t_bbox[3]] = 1 
        for bbox in bboxes[1:]: 
            weights[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1 
        
        train_target = train_target.flatten() # "fake" it as batched data to use weights 
        weights = weights.reshape(-1,1) 

        in_emb = extractor(img) 
        h, o, h_c, o_c = lwn(in_emb, h, o, h_c, o_c) 
        out = expander(o, to_img_shape=False) 

        loss = empty_bceloss(out, torch.zeros_like(out)) @ weights # since it's all supposed to be 0. 
        loss.backward(retain_graph=True) # not updating weights yet, just adding to gradients 
        
        losses.append(loss.item())
        del loss 
        
        torch.cuda.empty_cache()

    optimizer.step() 

    print("EPISODE AVG LOSS:", sum(losses)/len(losses)) 
    print("PREDICTION AVG LOSS:", sum(losses[-target_display:])/target_display)




