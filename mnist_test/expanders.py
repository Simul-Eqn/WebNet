import torch 
import torch.nn as nn 

class LinearsExpander(nn.Module): 
    def __init__(self, in_size:int, hidden_sizes:list[int], out_shape:tuple[int], activations=nn.Sigmoid() ): 

        nn.Module.__init__(self) 

        # activations can be a list 
        out_size = 1 
        for s in out_shape: out_size *= s 

        if len(hidden_sizes)==0: 
            # just one layer? 
            self.layers = [nn.Linear(in_size, out_size)] 
        else: 
            self.layers = [nn.Linear(in_size, hidden_sizes[0])] 
            for i in range(len(hidden_sizes)-1): 
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1])) 
            self.layers.append(nn.Linear(hidden_sizes[-1], out_size)) 
        
        self.in_size = in_size 
        self.hidden_sizes = hidden_sizes 
        self.out_size = out_size 
        self.out_shape = out_shape 
        self.activations = activations 
    
    def forward(self, x, to_img_shape=False): 
        if type(self.activations) is list: 
            for lidx in range(len(self.layers)): 
                x = self.layers[lidx](x) 
                x = self.activations[lidx](x) 
        else: 
            for layer in self.layers: 
                x = layer(x) 
                x = self.activations(x) 
        out_shape = list(x.shape)[:-1] + list(self.out_shape) # in case it's batched 

        if to_img_shape: 
            return x.reshape(out_shape) 
        return x 

            



