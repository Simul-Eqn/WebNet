import torch 
import torch.nn as nn 

from concurrent.futures import ThreadPoolExecutor 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# TODO: Is some form of attention useful? 

class WebNet(nn.Module):
    normalize = True 
    def __init__(self, n_input_nodes:int, n_hidden_nodes:int, n_output_nodes:int, 
                 update_fn = lambda v, inv: inv, aggregate_fn = lambda fromi,fromh: fromi+fromh, #h_has_self_weights=True, 
                 device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')): 
        # output_nodes must be sorted, 0-indexed. 
        # update_fn decides how to update. Default means it doesn't retain past info and it 
        #   just becomes the input, which makes this like an MLP. A custom input could make it 
        #   have an activation function like nn.ReLU()(inv), but it offers more flexibility. 
        # h_has_self_weights decides if h_to_h diagonal is all 0 or has weights. True means retains info. 
        super(WebNet, self).__init__() 

        self.device = device 

        self.n_input_nodes = n_input_nodes 
        self.n_hidden_nodes = n_hidden_nodes 
        self.n_output_nodes = n_output_nodes 
        self.update_fn = update_fn 
        self.aggregate_fn = aggregate_fn 
        #self.h_has_self_weights = h_has_self_weights 

        # TODO: initialize it differently to work better e.g. similar to linear RNN https://arxiv.org/abs/2303.06349 
        # this makes it avoid vanishing/exploding gradient problem 
        # normalize weights or smtg perhaps too 
        self.i_to_h = nn.Linear(n_input_nodes, n_hidden_nodes, device=device) #nn.Parameter(torch.rand((hidden_nodes, input_nodes))) 
        self.h_to_h = nn.Linear(n_hidden_nodes, n_hidden_nodes, device=device) #nn.Parameter(torch.rand((hidden_nodes, hidden_nodes))) 
        self.i_to_o = nn.Linear(n_input_nodes, n_output_nodes, device=device) #nn.Parameter(torch.rand((output_nodes, input_nodes))) 
        self.h_to_o = nn.Linear(n_hidden_nodes, n_output_nodes, device=device) #nn.Parameter(torch.rand((output_nodes, hidden_nodes)))

        #self.check_self_weights() 
        if WebNet.normalize: self.normalize_weights() 

    '''def check_self_weights(self): 
        if not self.h_has_self_weights: 
            # set h_to_h diagonal to None 
            self.h_to_h.data.fill_diagonal_(0) 
            if WebNet.normalize: self.normalize_weights() 
    '''

    def normalize_weights(self): # can force normalize if wanted 
        # TODO: NORMALIZE WEIGHTS 
        pass 


    def forward(self, i:torch.Tensor, h:torch.Tensor, o:torch.Tensor):
        # i=None just means no input 
        if i is None: 
            inh = self.h_to_h(h) 
            ino = self.h_to_o(h) 
        else: 
            inh = self.aggregate_fn( (self.i_to_h(i)) , (self.h_to_h(h)) ) # input into h 
            ino = self.aggregate_fn((self.i_to_o(i)) , (self.h_to_o(h)) ) # input into o 

        return self.update_fn(h, inh), self.update_fn(o, ino)

# input size: (?, total_nodes, 1) where ? is the batch size.


class LSTMWebNet(nn.Module):
    normalize = False 

    def __init__(self, n_input_nodes:int, n_hidden_nodes:int, n_output_nodes:int, #h_has_self_weights=False, 
                 batch_first=True, h_cell_state_size=None, o_cell_state_size=None, cell_state_size=4, 
                 h_share_lstm=True, o_share_lstm=True, all_share_lstm=False, 
                 in_aggregation = lambda from_i, from_h: from_i + from_h, in_activation=nn.ReLU(), lstm_activation=nn.ReLU(), 
                 device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), ): 
        # output_nodes must be sorted, 0-indexed. 
        # if all the _share_lstm == False, then h_cell and o_cell default to cell 
        # if all_share_lstm == True, then all is just 1 lstm no separate h_cell and o_cell, just cell 
        # NOTE: if any _share_lstm == True, consider setting lstm learning rate  a lot lower, since it's backpropagated (n_hidden_hodes + n_output_nodes)
        super(LSTMWebNet, self).__init__() 

        self.device = device 

        if (h_cell_state_size is None): 
            if (not h_share_lstm) and (not all_share_lstm) : 
                print("WARNING: (not h_share_lstm) and (not all_share_lstm), but h_cell_state_size is None. Setting to cell_state_size")
            h_cell_state_size = cell_state_size 
        if (o_cell_state_size is None): 
            if (not o_share_lstm) and (not all_share_lstm): 
                print("WARNING: (not o_share_lstm) and (not all_share_lstm), but o_cell_state_size is None. Setting to cell_state_size")
            o_cell_state_size = cell_state_size 

        self.n_input_nodes = n_input_nodes 
        self.n_hidden_nodes = n_hidden_nodes 
        self.n_output_nodes = n_output_nodes 
        self.in_aggregation = in_aggregation 
        self.in_activation = in_activation 
        self.lstm_activation = lstm_activation 
        #self.h_has_self_weights = h_has_self_weights 

        self.batch_first = batch_first 

        
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

        # TODO: is rand the best way to initialize parameters? 
        self.i_to_h = nn.Linear(n_input_nodes, n_hidden_nodes, device=device) #nn.Parameter(torch.rand((hidden_nodes, input_nodes))) 
        self.h_to_h = nn.Linear(n_hidden_nodes, n_hidden_nodes, device=device) #nn.Parameter(torch.rand((hidden_nodes, hidden_nodes))) 
        self.i_to_o = nn.Linear(n_input_nodes, n_output_nodes, device=device) #nn.Parameter(torch.rand((output_nodes, input_nodes))) 
        self.h_to_o = nn.Linear(n_hidden_nodes, n_output_nodes, device=device) #nn.Parameter(torch.rand((output_nodes, hidden_nodes)))

        #self.check_self_weights() # only works if h_has_self_weights = False 
        if LSTMWebNet.normalize: self.normalize_weights() 

        if all_share_lstm: 
            self.lstm = nn.LSTM(1, cell_state_size, proj_size=1, device=device)
        else: 
            if h_share_lstm: 
                self.h_lstm = nn.LSTM(1, h_cell_state_size, proj_size=1, device=device) 
            else: 
                self.hs = nn.ModuleList([nn.LSTM(1, h_cell_state_size, proj_size=1, device=device) for _ in range(n_hidden_nodes)]) 
            
            if o_share_lstm: 
                self.o_lstm = nn.LSTM(1, o_cell_state_size, proj_size=1, device=device)
            else: 
                self.os = nn.ModuleList([nn.LSTM(1, o_cell_state_size, proj_size=1, device=device) for _ in range(n_output_nodes)]) 
    
    def get_non_lstm_params(self): 
        def params_generator(): 
            params = self.i_to_h.parameters() 
            while True: 
                try: 
                    yield next(params) 
                except Exception: 
                    break 
            
            params = self.i_to_o.parameters() 
            while True: 
                try: 
                    yield next(params) 
                except Exception: 
                    break 
            
            params = self.h_to_h.parameters() 
            while True: 
                try: 
                    yield next(params) 
                except Exception: 
                    break 
            
            params = self.h_to_o.parameters() 
            while True: 
                try: 
                    yield next(params) 
                except Exception: 
                    break 
        
        return params_generator() 

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


    '''
    def check_self_weights(self): 
        if not self.h_has_self_weights: 
            # set h_to_h diagonal to None 
            self.h_to_h.data.fill_diagonal_(0) 
            if LSTMWebNet.normalize: self.normalize_weights() 
    '''

    def normalize_weights(self): # can force normalize if wanted 
        # TODO: NORMALIZE WEIGHTS 
        pass 


    def get_h_cell_shape(self, batch_size:int=None): 
        # batch_size = None means unbatched, else it's batched. 
        if (batch_size is None): 
            return (1, self.h_cell_state_size) 
        return (1, batch_size, self.h_cell_state_size) 
    
    def h_cells_zeros(self, batch_size:int=None, device=device): 
        s = self.get_h_cell_shape(batch_size) 
        return [torch.zeros(s).to(device) for _ in range(self.n_hidden_nodes)] 
    

    def get_o_cell_shape(self, batch_size:int=None): 
        # batch_size = None means unbatched, else it's batched. 
        if (batch_size is None): 
            return (1, self.o_cell_state_size) 
        return (1, batch_size, self.o_cell_state_size) 
    
    def o_cells_zeros(self, batch_size:int=None, device=device): 
        s = self.get_o_cell_shape(batch_size) 
        return [torch.zeros(s).to(device) for _ in range(self.n_output_nodes)] 
    
    #                  input values    hidden values  output values 
    def forward(self, i:torch.Tensor, h:torch.Tensor, o:torch.Tensor, 
                h_cells:list[torch.Tensor], o_cells:list[torch.Tensor]):
        #print(i.device) 
        #print(h.device) 
        #print(o.device) 

        # i=None just means no input 
        if i is None: 
            inh = self.h_to_h(h) 
            ino = self.h_to_o(h) 
        else: 
            inh = self.in_aggregation( (self.i_to_h(i)) , (self.h_to_h(h)) ) # input into h 
            ino = self.in_aggregation( (self.i_to_o(i)) , (self.h_to_o(h)) ) # input into o 

        #print(ino.device) 
        #print(inh.device) 

        inh = self.in_activation(inh) # input to hiddens 
        ino = self.in_activation(ino) # input to outputs 

        #print(inh) 
        #print("INH SHAPE:", inh.shape)
        #print(ino) 
        #print("INO SHAPE:", ino.shape) 

        batched = len(h.shape)>1 
        
        # run the input through the hiddens and outputs' lstms 
        if self.all_share_lstm: lstm = self.lstm 
        else: lstm=None
        if batched: 
            # settle hiddens 
            if self.h_share_lstm: lstm = self.h_lstm 
            h_outs = [] 
            h_cell_outs = [] 

            def settle_hidx(hidx, lstm): 
                if (not self.all_share_lstm) and (not self.h_share_lstm): 
                    lstm = self.hs[hidx]

                h_out, (_, h_cell_out) = lstm(inh[:, hidx:hidx+1].reshape((1,-1,1)), 
                                (h[:, hidx:hidx+1].reshape((1,-1,1)), h_cells[hidx])) 
                
                if self.batch_first: 
                    h_outs.append(h_out.reshape((-1,1))) 
                else: 
                    h_outs.append(h_out.reshape((1,-1))) 
                h_cell_outs.append(h_cell_out) 
            
            with ThreadPoolExecutor() as executor: 
                for hidx in range(self.n_hidden_nodes): 
                    executor.submit(settle_hidx, hidx, lstm)
            
            '''
            for hidx in range(self.n_hidden_nodes): 
                if (not self.all_share_lstm) and (not self.h_share_lstm): 
                    lstm = self.hs[hidx]

                h_out, (_, h_cell_out) = lstm(inh[:, hidx:hidx+1].reshape((1,-1,1)), 
                                (h[:, hidx:hidx+1].reshape((1,-1,1)), h_cells[hidx])) 
                if self.batch_first: 
                    h_outs.append(h_out.reshape((-1,1))) 
                else: 
                    h_outs.append(h_out.reshape((1,-1))) 
                h_cell_outs.append(h_cell_out) 
            ''' 
            
            if self.batch_first: 
                new_h = torch.cat(h_outs, dim=1) 
            else: 
                new_h = torch.cat(h_outs, dim=0)
            
            
            # settle outs 
            if self.o_share_lstm: lstm = self.o_lstm 
            o_outs = [] 
            o_cell_outs = [] 

            def setle_oidx(oidx, lstm): 
                if (not self.all_share_lstm) and (not self.o_share_lstm): 
                    lstm = self.os[oidx]

                o_out, (_, o_cell_out) = lstm(ino[:, oidx:oidx+1].reshape((1,-1,1)), 
                                (o[:, oidx:oidx+1].reshape((1,-1,1)), o_cells[oidx]))
                if self.batch_first: 
                    o_outs.append(o_out.reshape((-1,1))) 
                else: 
                    o_outs.append(o_out.reshape((1,-1))) 
                o_cell_outs.append(o_cell_out) 

            with ThreadPoolExecutor() as executor: 
                for oidx in range(self.n_output_nodes): 
                    executor.submit(settle_oidx, oidx, lstm)

            if self.batch_first: 
                new_o = torch.cat(o_outs, dim=1) 
            else: 
                new_o = torch.cat(o_outs, dim=0)

            
        else: 
            # settle hiddens 
            if self.h_share_lstm: lstm = self.h_lstm 
            h_outs = [] 
            h_cell_outs = [] 
            
            def settle_hidx(hidx, lstm): 
                if (not self.all_share_lstm) and (not self.h_share_lstm): 
                    lstm = self.hs[hidx]

                h_out, (_, h_cell_out) = lstm(inh[hidx:hidx+1].reshape((1,1)), 
                                (h[hidx:hidx+1].reshape((1,1)), h_cells[hidx])) 
                h_outs.append(h_out.reshape(1)) 
                h_cell_outs.append(h_cell_out) 

            with ThreadPoolExecutor() as executor: 
                for hidx in range(self.n_hidden_nodes): 
                    executor.submit(settle_hidx, hidx, lstm) 

            new_h = torch.cat(h_outs, dim=0) 

            
            # settle outs 
            if self.o_share_lstm: lstm = self.o_lstm 
            o_outs = [] 
            o_cell_outs = [] 
            
            def settle_oidx(oidx, lstm): 
                if (not self.all_share_lstm) and (not self.o_share_lstm): 
                    lstm = self.os[oidx]

                o_out, (_, o_cell_out) = lstm(ino[oidx:oidx+1].reshape((1,1)), 
                                (o[oidx:oidx+1].reshape((1,1)), o_cells[oidx])) 
                o_outs.append(o_out.reshape(1)) 
                o_cell_outs.append(o_cell_out) 

            with ThreadPoolExecutor() as executor: 
                for oidx in range(self.n_output_nodes): 
                    executor.submit(settle_oidx, oidx, lstm)
            
            new_o = torch.cat(o_outs, dim=0) 
            

        #     input to hiddens at next step; input to next step outputs; hidden memory; output memory 
        return self.lstm_activation(new_h), self.lstm_activation(new_o), h_cell_outs, o_cell_outs  



if __name__ == "__main__": 

    batch_size = 3 
    inputs = 10 
    hiddens = 88 
    outputs = 2 

    wn = WebNet(inputs, hiddens, outputs)

    i0 = torch.rand((batch_size, inputs)) # this just for testing 
    h = torch.zeros((batch_size, hiddens)) 
    o = torch.zeros((batch_size, outputs)) 

    lwn = LSTMWebNet(inputs, hiddens, outputs) 
    r = lwn(i0, h, o, lwn.h_cells_zeros(3), lwn.o_cells_zeros(3)) 


