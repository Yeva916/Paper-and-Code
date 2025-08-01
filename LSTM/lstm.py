import torch
import torch.nn as nn

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