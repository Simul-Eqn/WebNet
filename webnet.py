import torch 
import torch.nn as nn 



class WebNet(nn.Module):
    def __init__(self, total_nodes:int, output_nodes:list, activation=nn.ReLU(), 
                 output_memory=0.95, ): 
        # output_nodes must be sorted, 0-indexed. 
        # output_memory means how much output node values gets retained 
        super(WebNet, self).__init__() 

        self.total_nodes = total_nodes 
        self.output_nodes = output_nodes 

        self.params = [] 
        output_nodes_idx = 0 
        for i in range(total_nodes): 
            if (i == output_nodes[output_nodes_idx]): 
                # since it's an output node, it doesn't affect other nodes 
                t = torch.zeros((total_nodes, 1)) 
                t[i][0] = output_memory 
                self.params.append(nn.Parameter(t, requires_grad=False))
                output_nodes_idx += 1 
            else: 
                self.params.append(nn.Parameter(torch.rand((total_nodes, 1)), requires_grad=True)) 

        self.activation = activation 
    
    def forward(self, x):
        if (len(x.shape)==3): n = x.shape[0]
        else: n = 1

        # pass through graph + activation fn 
        x = torch.cat(self.params, dim=1) @ x 
        x = self.activation(x)
        
        # batch normalization: 
        x = self.total_nodes*x/(x.sum()/n) 
        return x 

# input size: (?, total_nodes, 1) where ? is the batch size.

batch_size = 3 
inputs = 30 
hiddens = 68 
outputs = 2 

wn = WebNet(inputs+hiddens+outputs, list(range(inputs+hiddens, inputs+hiddens+outputs)))

x = torch.cat([torch.rand((batch_size, inputs, 1)) , torch.zeros((3, hiddens+outputs, 1)) ], dim=1) 

