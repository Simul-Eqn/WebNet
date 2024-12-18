from webnet import LSTMWebNet 
from webnet_lstm_multithread import LSTMWebNet as MLSTMWebNet
import timeit 

import torch 
device = torch.device('cuda') 

n_inputs = 1000 
n_hiddens = 3000 
n_outputs = 1000 
in_emb = torch.rand((1000)).cuda() 

# normal 
lwn = LSTMWebNet(n_inputs, n_hiddens, n_outputs, device=device).to(device) 
h = torch.zeros((n_hiddens)).to(device) 
o = torch.zeros((n_outputs)).to(device) 
h_c = lwn.h_cells_zeros(device=device)
o_c = lwn.o_cells_zeros(device=device)

with torch.no_grad(): 
    lwn(in_emb, h, o, h_c, o_c) 

# with muiltithreading 
mlwn = MLSTMWebNet(n_inputs, n_hiddens, n_outputs, device=device).to(device) 
mh = torch.zeros((n_hiddens)).to(device) 
mo = torch.zeros((n_outputs)).to(device) 
mh_c = mlwn.h_cells_zeros(device=device)
mo_c = mlwn.o_cells_zeros(device=device)

with torch.no_grad(): 
    mlwn(in_emb, mh, mo, mh_c, mo_c) 

with torch.no_grad(): 
    print("NO MULTITHREADING:", timeit.timeit("lwn(in_emb, h, o, h_c, o_c) ", globals=globals(), number=100)/100) 
    print("MULTITHREADING:", timeit.timeit("mlwn(in_emb, mh, mo, mh_c, mo_c) ", globals=globals(), number=100)/100) 
'''
output on my computer: 
NO MULTITHREADING: 0.40816579800099134
MULTITHREADING: 0.5550997649994679
'''
