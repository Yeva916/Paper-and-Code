import torch
import torch.nn as nn
from typing import Optional, Tuple

class LSTMCell(nn.Module):
    def __init__(self,input_size:int,hidden_size:int,layer_norm:bool):
        super().__init__()
        self.hidden_lin = nn.Linear(hidden_size ,hidden_size * 4)
        self.input_lin = nn.Linear(input_size,input_size * 4)

        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c = nn.Identity()
        
    def forward(self,x:torch.tensor,h:torch.tensor,c:torch.tensor):

        ifgo = self.hidden_lin(h) + self.input_lin(x)

        ifgo = ifgo.chunk(4,dim=-1)

        ifgo = [self.layer_norm[i](ifgo[i]) for i in range(4)]

        i,f,g,o = ifgo

        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return c_next,h_next

model = LSTMCell(input_size=3,hidden_size=3,layer_norm=True)
a = torch.tensor([1,2,3],dtype=torch.float32)
c_next,h_next = model(a,a,a)
print({"c_next":c_next.detach(),
       "h_next":h_next.detach()
       })

'''
output ->
{'c_next': tensor([-0.4601,  1.1984,  2.5017]), 'h_next': tensor([-0.6833,  0.0275,  0.3205])}

'''

class LSTM(nn.Module):
    def __init__(self,input_size:int,hidden_size:int,n_layers:int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cells = nn.ModuleList([LSTMCell(input_size,hidden_size)] +
                                   [LSTMCell(input_size,hidden_size) for _ in range(self.n_layer - 1)])

    def forward(self,x:torch.Tensor,state:Optional[Tuple[torch.Tensor,torch.Tensor]]):

        # x = [num_steps,batch_size,input_size]

        num_steps,batch_size = x[:2]

        if state is None:
            h = [x.new_zeros(batch_size,self.hidden_size) for _ in range(self.n_layer)]
            c = [x.new_zeros(batch_size,self.hidden_size) for _ in range(self.n_layer)]
        else:
            h,c = state
        
        h,c = list(h),list(c)

        out = []

        for t in range(num_steps):

            inp = x[t]

            for layer in self.n_layers:
                h[layer],c[layer] = self.cells[layer](inp,h[layer],c[layer])

                inp = h[layer]

            out.append(h[-1])
        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.state(c)
        return out,(h,c)