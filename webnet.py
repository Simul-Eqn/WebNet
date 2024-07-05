import torch 
import torch.nn as nn 

# TODO: Is some form of attention useful? 

class WebNet(nn.Module):
    normalize = True 
    def __init__(self, input_nodes:int, hidden_nodes:int, output_nodes:int, 
                 update_fn = lambda v, inv: inv, h_has_self_weights=True): 
        # output_nodes must be sorted, 0-indexed. 
        # update_fn decides how to update. Default means it doesn't retain past info and it 
        #   just becomes the input, which makes this like an MLP. A custom input could make it 
        #   have an activation function like nn.ReLU()(inv), but it offers more flexibility. 
        # h_has_self_weights decides if h_to_h diagonal is all 0 or has weights. True means retains info. 
        super(WebNet, self).__init__() 

        self.input_nodes = input_nodes 
        self.hidden_nodes = hidden_nodes 
        self.output_nodes = output_nodes 
        self.update_fn = update_fn 
        self.h_has_self_weights = h_has_self_weights 

        # TODO: initialize it differently to work better e.g. similar to linear RNN https://arxiv.org/abs/2303.06349 
        # this makes it avoid vanishing/exploding gradient problem 
        # normalize weights or smtg perhaps too 
        self.i_to_h = nn.Parameter(torch.rand((hidden_nodes, input_nodes))) 
        self.h_to_h = nn.Parameter(torch.rand((hidden_nodes, hidden_nodes))) 
        self.i_to_o = nn.Parameter(torch.rand((output_nodes, input_nodes))) 
        self.h_to_o = nn.Parameter(torch.rand((output_nodes, hidden_nodes)))

        self.check_self_weights() 
        if WebNet.normalize: self.normalize_weights() 

    def check_self_weights(self): 
        if not self.h_has_self_weights: 
            # set h_to_h diagonal to None 
            self.h_to_h.data.fill_diagonal_(0) 
            if WebNet.normalize: self.normalize_weights() 
    
    def normalize_weights(self): # can force normalize if wanted 
        # TODO: NORMALIZE WEIGHTS 
        pass 


    def forward(self, i:torch.Tensor, h:torch.Tensor, o:torch.Tensor):
        # i=None just means no input 
        if i is None: 
            inh = self.h_to_h @ h 
            ino = self.h_to_o @ o 
        else: 
            inh = (self.i_to_h @ i) + (self.h_to_h @ h) # input into h 
            ino = (self.i_to_o @ i) + (self.h_to_o @ h) # input into o 

        return self.update_fn(h, inh), self.update_fn(o, ino)

# input size: (?, total_nodes, 1) where ? is the batch size.


class LSTMWebNet(nn.Module):
    normalize = False 

    def __init__(self, n_input_nodes:int, n_hidden_nodes:int, n_output_nodes:int, h_has_self_weights=False, 
                 batch_first=True, h_cell_state_size=None, o_cell_state_size=None, cell_state_size=4, 
                 h_share_lstm=True, o_share_lstm=True, all_share_lstm=False, 
                 in_aggregation = lambda from_i, from_h: from_i + from_h, in_activation=nn.ReLU(), lstm_activation=nn.ReLU() ): 
        # output_nodes must be sorted, 0-indexed. 
        # if all the _share_lstm == False, then h_cell and o_cell default to cell 
        # if all_share_lstm == True, then all is just 1 lstm no separate h_cell and o_cell, just cell 
        # NOTE: if any _share_lstm == True, consider setting lstm learning rate  a lot lower, since it's backpropagated (n_hidden_hodes + n_output_nodes)
        super(LSTMWebNet, self).__init__() 

        if (not h_share_lstm) and (not all_share_lstm) and (h_cell_state_size is None): 
            h_cell_state_size = cell_state_size 
        if (not o_share_lstm) and (not all_share_lstm) and (o_cell_state_size is None): 
            o_cell_state_size = cell_state_size 

        self.n_input_nodes = n_input_nodes 
        self.n_hidden_nodes = n_hidden_nodes 
        self.n_output_nodes = n_output_nodes 
        self.in_aggregation = in_aggregation 
        self.in_activation = in_activation 
        self.lstm_activation = lstm_activation 
        self.h_has_self_weights = h_has_self_weights 

        
        self.all_share_lstm = all_share_lstm 
        if all_share_lstm: 
            self.cell_state_size = cell_state_size 
            self.h_cell_state_size = cell_state_size 
            self.o_cell_state_size = cell_state_size
            self.h_share_lstm = False 
            self.o_share_lstm = False 
        else: 
            self.h_share_lstm = h_share_lstm 
            self.o_share_lstm = o_share_lstm 
            self.h_cell_state_size = h_cell_state_size 
            self.o_cell_state_size = o_cell_state_size
        self.batch_first = batch_first 

        # TODO: is rand the best way to initialize parameters? 
        self.i_to_h = nn.Parameter(torch.rand((n_hidden_nodes, n_input_nodes))) 
        self.h_to_h = nn.Parameter(torch.rand((n_hidden_nodes, n_hidden_nodes))) 
        self.i_to_o = nn.Parameter(torch.rand((n_output_nodes, n_input_nodes))) 
        self.h_to_o = nn.Parameter(torch.rand((n_output_nodes, n_hidden_nodes)))

        self.check_self_weights() # only works if h_has_self_weights = False 
        if LSTMWebNet.normalize: self.normalize_weights() 

        if all_share_lstm: 
            self.lstm = nn.LSTM(1, cell_state_size, batch_first=batch_first, proj_size=1)
        else: 
            if h_share_lstm: 
                self.h_lstm = nn.LSTM(1, h_cell_state_size, batch_first=batch_first, proj_size=1) 
            else: 
                self.hs = [nn.LSTM(1, h_cell_state_size, batch_first=batch_first, proj_size=1) for _ in range(n_hidden_nodes)] 
            
            if o_share_lstm: 
                self.o_lstm = nn.LSTM(1, o_cell_state_size, batch_first=batch_first, proj_size=1)
            else: 
                self.os = [nn.LSTM(1, o_cell_state_size, batch_first=batch_first, proj_size=1) for _ in range(n_output_nodes)] 

    def get_h_lstm_params(self): 
        if self.all_share_lstm: 
            return self.lstm.parameters() 
        
        if self.h_share_lstm: 
            return self.h_lstm.parameters() 
            
        def hs_generator(): 
            for hidx in range(len(self.hs)): 
                params = self.hs[hidx].parameters() 
                while True: 
                    try: 
                        yield next(params) 
                    except Exception: 
                        break 
        return hs_generator() 
    
    def get_o_lstm_params(self): 
        if self.all_share_lstm: 
            return self.lstm.parameters()
        
        if self.o_share_lstm: 
            return self.o_lstm.parameters() 

        def os_generator(): 
            for oidx in range(len(self.os)): 
                params = self.os[oidx].parameters() 
                while True: 
                    try: 
                        yield next(params) 
                    except Exception: 
                        break  
        return os_generator() 

    def get_lstm_params(self): 
        if self.all_share_lstm: 
            return self.lstm.parameters() 

        def hs_os_generator(): 
            hs_gen = self.get_h_lstm_params() 
            while True: 
                try: 
                    yield next(hs_gen) 
                except Exception: 
                    break 
            
            os_gen = self.get_o_lstm_params() 
            while True: 
                try: 
                    yield next(os_gen) 
                except Exception: 
                    break 

        return hs_os_generator() 

    
    def check_self_weights(self): 
        if not self.h_has_self_weights: 
            # set h_to_h diagonal to None 
            self.h_to_h.data.fill_diagonal_(0) 
            if LSTMWebNet.normalize: self.normalize_weights() 
    
    def normalize_weights(self): # can force normalize if wanted 
        # TODO: NORMALIZE WEIGHTS 
        pass 


    def get_h_cell_shape(self, batch_size:int=None): 
        # batch_size = None means unbatched, else it's batched. 
        if (batch_size == None): 
            return (1, self.h_cell_state_size) 
        return (1, batch_size, self.h_cell_state_size) 
    
    def h_cells_zeros(self, batch_size:int=None): 
        s = self.get_h_cell_shape(batch_size) 
        return [torch.zeros(s) for _ in range(self.n_hidden_nodes)] 
    

    def get_o_cell_shape(self, batch_size:int=None): 
        # batch_size = None means unbatched, else it's batched. 
        if (batch_size == None): 
            return (1, self.o_cell_state_size) 
        return (1, batch_size, self.o_cell_state_size) 
    
    def o_cells_zeros(self, batch_size:int=None): 
        s = self.get_o_cell_shape(batch_size) 
        return [torch.zeros(s) for _ in range(self.n_output_nodes)] 
    
    #                  input values    hidden values  output values 
    def forward(self, i:torch.Tensor, h:torch.Tensor, o:torch.Tensor, 
                h_cells:list[torch.Tensor], o_cells:list[torch.Tensor]):
        # i=None just means no input 
        if i is None: 
            inh = self.h_to_h @ h 
            ino = self.h_to_o @ o 
        else: 
            inh = self.in_aggregation( (self.i_to_h @ i) , (self.h_to_h @ h) ) # input into h 
            ino = self.in_aggregation( (self.i_to_o @ i) , (self.h_to_o @ h) ) # input into o 

        inh = self.in_activation(inh) # input to hiddens 
        ino = self.in_activation(ino) # input to outputs 

        batched = len(i.shape)>2 
        
        # run the input through the hiddens and outputs' lstms 
        if self.all_share_lstm: lstm = self.lstm 
        if batched: 
            if self.batch_first: 
                # settle hiddens 
                if self.h_share_lstm: lstm = self.h_lstm 
                h_outs = [] 
                h_cell_outs = [] 
                for hidx in range(self.n_hidden_nodes): 
                    if (not self.all_share_lstm) and (not self.h_share_lstm): 
                        lstm = self.hs[hidx]

                    h_out, (_, h_cell_out) = lstm(inh[:, hidx:hidx+1, :].transpose(1,2), 
                                    (h[:, hidx:hidx+1, :].transpose(0,2).transpose(1,2), h_cells[hidx])) 
                    h_outs.append(h_out.transpose(1,2)) 
                    h_cell_outs.append(h_cell_out) 
                new_h = torch.cat(h_outs, dim=1) 
                
                # settle outs 
                if self.o_share_lstm: lstm = self.o_lstm 
                o_outs = [] 
                o_cell_outs = [] 
                for oidx in range(self.n_output_nodes): 
                    if (not self.all_share_lstm) and (not self.o_share_lstm): 
                        lstm = self.os[oidx]

                    o_out, (_, o_cell_out) = lstm(ino[:, oidx:oidx+1, :].transpose(1,2), 
                                    (o[:, oidx:oidx+1, :].transpose(0,2).transpose(1,2), o_cells[oidx])) 
                    o_outs.append(o_out.transpose(1,2)) 
                    o_cell_outs.append(o_cell_out) 
                new_o = torch.cat(o_outs, dim=1) 

            else: 
                # settle hiddens 
                if self.h_share_lstm: lstm = self.h_lstm 
                h_outs = [] 
                h_cell_outs = [] 
                for hidx in range(self.n_hidden_nodes): 
                    if (not self.all_share_lstm) and (not self.h_share_lstm): 
                        lstm = self.hs[hidx]

                    h_out, (_, h_cell_out) = lstm(inh[hidx:hidx+1, :, :].transpose(1,2), 
                                    (h[hidx:hidx+1, :, :].transpose(1,2), h_cells[hidx])) 
                    h_outs.append(h_out.transpose(1,2)) 
                    h_cell_outs.append(h_cell_out) 
                new_h = torch.cat(h_outs, dim=1) 
                
                # settle outs 
                if self.o_share_lstm: lstm = self.o_lstm 
                o_outs = [] 
                o_cell_outs = [] 
                for oidx in range(self.n_output_nodes): 
                    if (not self.all_share_lstm) and (not self.o_share_lstm): 
                        lstm = self.os[oidx]

                    o_out, (_, o_cell_out) = lstm(ino[oidx:oidx+1, :, :].transpose(1,2), 
                                    (o[oidx:oidx+1, :, :].transpose(1,2), o_cells[oidx])) 
                    o_outs.append(o_out.transpose(1,2)) 
                    o_cell_outs.append(o_cell_out) 
                new_o = torch.cat(o_outs, dim=1) 
        
        else: 
            # settle hiddens 
            if self.h_share_lstm: lstm = self.h_lstm 
            h_outs = [] 
            h_cell_outs = [] 
            for hidx in range(self.n_hidden_nodes): 
                if (not self.all_share_lstm) and (not self.h_share_lstm): 
                    lstm = self.hs[hidx]

                h_out, (_, h_cell_out) = lstm(inh[hidx:hidx+1, :].transpose(0,1), 
                                (h[hidx:hidx+1, :].transpose(0,1), h_cells[hidx])) 
                h_outs.append(h_out.transpose(0,1)) 
                h_cell_outs.append(h_cell_out) 
            new_h = torch.cat(h_outs, dim=0) 
            
            # settle outs 
            if self.o_share_lstm: lstm = self.o_lstm 
            o_outs = [] 
            o_cell_outs = [] 
            for oidx in range(self.n_output_nodes): 
                if (not self.all_share_lstm) and (not self.o_share_lstm): 
                    lstm = self.os[oidx]

                o_out, (_, o_cell_out) = lstm(ino[oidx:oidx+1, :].transpose(0,1), 
                                (o[oidx:oidx+1, :].transpose(0,1), o_cells[oidx])) 
                o_outs.append(o_out.transpose(0,1)) 
                o_cell_outs.append(o_cell_out) 
            new_o = torch.cat(o_outs, dim=0) 
            

        #     input to hiddens at next step; input to next step outputs; hidden memory; output memory 
        return self.lstm_activation(new_h), self.lstm_activation(new_o), h_cell_outs, o_cell_outs  



if __name__ == "__main__": 

    batch_size = 3 
    inputs = 10 
    hiddens = 88 
    outputs = 2 

    wn = WebNet(inputs, hiddens, outputs)

    i0 = torch.rand((batch_size, inputs, 1)) # this just for testing 
    h = torch.zeros((batch_size, hiddens, 1)) 
    o = torch.zeros((batch_size, outputs, 1)) 

