
import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F


class StyleHallucination(nn.Module):
    '''
    Style Hallucination Module.
    Reference:
      Zhao et al. Style-Hallucinated Dual Consistency Learning for Domain Generalized Semantic Segmentation. ECCV 2022.
      https://arxiv.org/pdf/2204.02548.pdf
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.concentration = torch.tensor([args.concentration_coeff]*args.base_style_num, device='cuda')
        self._dirichlet = tdist.dirichlet.Dirichlet(concentration=self.concentration)
        
        self.register_buffer("proto_mean", torch.zeros((args.base_style_num,args.style_dim), requires_grad=False))
        self.register_buffer("proto_std", torch.zeros((args.base_style_num,args.style_dim), requires_grad=False))


    def forward(self, x):
        B,C,H,W = x.size()
        x_mean = x.mean(dim=[2,3], keepdim=True) # B,C,1,1
        x_std = x.std(dim=[2,3], keepdim=True) + 1e-7 # B,C,1,1
        x_mean, x_std = x_mean.detach(), x_std.detach()

        x_norm = (x - x_mean) / x_std

        combine_weights = self._dirichlet.sample((B,)) # B,C
        combine_weights = combine_weights.detach()
        
        new_mean = combine_weights @ self.proto_mean.data # B,C
        new_std = combine_weights @ self.proto_std.data 

        x_new = x_norm * new_std.unsqueeze(-1).unsqueeze(-1) + new_mean.unsqueeze(-1).unsqueeze(-1)

        return x, x_new



