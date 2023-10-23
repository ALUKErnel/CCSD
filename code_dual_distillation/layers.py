

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter



class MultiHeadGraphAttention_weight(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, concat_dropout, leaky_alpha):
        super(MultiHeadGraphAttention_weight, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.attn_dropout = attn_dropout
        self.concat_dropout = concat_dropout
        self.leaky_alpha = leaky_alpha
        
        
        self.w = Parameter(torch.Tensor(self.n_head, self.f_in, self.f_out))
        self.bias = Parameter(torch.Tensor(self.f_out))
        self.a_src = Parameter(torch.Tensor(self.n_head, self.f_out, 1))
        self.a_dst = Parameter(torch.Tensor(self.n_head, self.f_out, 1))
        

        self.leaky_relu = nn.LeakyReLU(negative_slope=self.leaky_alpha)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        
        
        assert self.f_in == self.n_head * self.f_out 


        init.xavier_uniform_(self.w)
        init.zeros_(self.bias.data)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)
        

    def forward(self, h, adj):
        n = h.size(0) # h is of size n x f_in (node feature)
        h_prime = torch.matmul(h.unsqueeze(0), self.w) #  n_head x n x f_out
        attn_src = torch.bmm(h_prime, self.a_src) # n_head x n x 1   # todo 
        attn_dst = torch.bmm(h_prime, self.a_dst) # n_head x n x 1   # todo 
        attn = attn_src.expand(-1, -1, n) + attn_dst.expand(-1, -1, n).permute(0, 2, 1) # n_head x n x n

        attn = self.leaky_relu(attn)
        # attn.data.masked_fill_(1 - adj, -999)    # -999
        zero_vec = -9e15*torch.ones_like(attn)
        attn = torch.where(adj > 0, attn, zero_vec)
        attn = self.softmax(attn) # n_head x n x n
        attn = self.dropout(attn) # todo 
        
        output = torch.bmm(attn, h_prime) # n_head x n x f_out
        
        output = output + self.bias 
        
        # multi-head concat 
        output = output.transpose(0,1).contiguous().view(n, -1) # n x n_head*f_out  so next f_in == n_head * f_out
        
        output = F.dropout(output, self.concat_dropout)
        
        weight = torch.sum(attn, dim=0) # n x n
        
        return output, weight 
        
        
        
        