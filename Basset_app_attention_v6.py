from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import copy

class PositionalEncoder(nn.Module):
    def __init__(self, d_model,sequence_length,maxpoolsize):
        super().__init__()
        max_seq_len = sequence_length//maxpoolsize
        self.d_model = d_model
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        return(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout = 0.3):
        super().__init__()
        # We set d_ff as a default to 2048
        self.d_ff = d_model
        self.linear_1 = nn.Linear(d_model, self.d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(self.d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        #x = self.linear_2(x)
        return(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return(norm)

def get_mapping(sequence_length,maxpoolsize):
    '''
    # mapping v1
    n_pos = sequence_length // maxpoolsize
    step_size = 32
    n_pos_class = sequence_length//step_size
    repeating_ratio = n_pos//n_pos_class
    pos_k = (n_pos_class*2)//3

    df = pd.DataFrame(columns=list(range(n_pos)),index=list(range(n_pos)))
    for j in range(n_pos):
        for i in range(n_pos):
            if i-j >=0:
                df.at[i,j] = -((i-j)//repeating_ratio)
            else:
                #df.at[i,j] = -df.at[j,i] # non-symmetric
                df.at[i,j] = df.at[j,i] # symmetric

    mapping = torch.LongTensor(df.values.tolist())
    mapping = torch.clamp(mapping,min = - pos_k, max = pos_k)
    mapping += pos_k


    # mapping v2
    n_pos = sequence_length // maxpoolsize
    log_base = 8
    pos_k = 8

    df = pd.DataFrame(columns=list(range(n_pos)),index=list(range(n_pos)))
    for j in range(n_pos):
        for i in range(n_pos):
            if i ==j:
                df.at[i,j] = 0
            elif i-j >0:
                df.at[i,j] = -int((np.log((i-j)*maxpoolsize)/np.log(log_base)))
            else:
                df.at[i,j] = df.at[j,i]

    mapping = torch.LongTensor(df.values.tolist())
    mapping = torch.clamp(mapping,min = - pos_k, max = pos_k)
    mapping += pos_k
    '''

    # mapping v3
    n_pos = sequence_length // maxpoolsize
    log_base = 2 # syn v7; in syn v6: 2
    pos_k = 8

    df = pd.DataFrame(columns=list(range(n_pos)),index=list(range(n_pos)))
    for j in range(n_pos):
        for i in range(n_pos):
            if i ==j:
                df.at[i,j] = 0
            elif i-j >0:
                df.at[i,j] = int((np.log((i-j)*maxpoolsize)/np.log(log_base)))
            else:
                df.at[i,j] = df.at[j,i]

    mapping = torch.LongTensor(df.values.tolist())
    mapping = torch.clamp(mapping,min = - pos_k, max = pos_k)
    mapping += pos_k

    mapping = mapping.view(1,n_pos,n_pos)
    mapping = torch.cat([mapping]*2)
    return(mapping)

def attention_rpos(q, k, v, rpos_k, rpos_v, d_k, mask=None, dropout=None):
    bs,h,npos,dk = q.size() # q: bs*h*npos*dk
    # rpos_k / rpos_v: npos*npos*h*dk

    # calculating the attention score
    # standard attention score
    standard_scores = torch.matmul(q, k.transpose(-2,-1)) /  math.sqrt(d_k) #: bs*h* npos*npos

    # positional attention score
    q = q.contiguous().view(bs,h,npos,1,dk) # bs,h,npos,1,dk
    rpos_k = rpos_k.view(1,npos,npos,h,dk)
    rpos_k = torch.cat([rpos_k]*bs) # bs,npos,npos,h,dk
    rpos_k = rpos_k.transpose(2,3) # bs,npos,h,npos,dk
    rpos_k = rpos_k.transpose(3,4) # bs,npos,h,dk,npos
    rpos_k = rpos_k.transpose(1,2) # bs,h,npos,dk,npos

    position_scores = torch.matmul(q,rpos_k) / math.sqrt(d_k) # bs,h,npos,npos
    position_scores = position_scores.contiguous().view(bs,h,npos,npos)

    scores = F.softmax(standard_scores + position_scores, dim=-1) # bs*h*npos*npos

    if dropout is not None:
        scores = dropout(scores)

    standard_output = torch.matmul(scores, v) / math.sqrt(d_k) # bs*h*npos*dk
    '''
    scores = scores.view(bs,h,npos,1,npos)
    rpos_v = rpos_v.view(1,npos,npos,h,dk)
    rpos_v = torch.cat([rpos_v]*bs) # bs,npos,npos,h,dk
    rpos_v = rpos_v.transpose(2,3) # bs,npos,h,npos,dk
    rpos_v = rpos_v.transpose(1,2)  # bs,h.npos,npos,dk

    position_output = torch.matmul(scores,rpos_v) / math.sqrt(d_k) #bs,h,npos,1,dk
    position_output = position_output.contiguous().view(bs,h,npos,dk)

    # scores: bs*h*npos*1*npos
    # (bs,npos,npos,h,dk) --> bs,h,npos,npos,dk
    # aim: bs*h*npos*dk
    return(standard_output + position_output)
    '''
    return(standard_output)

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) #scaled dot product

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return(output)

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_input, d_output, dropout = 0.3):
        super().__init__()

        self.d_input = d_input
        self.d_output = d_output
        self.d_k = d_output // heads
        self.h = heads

        self.q_linear = nn.Linear(d_input, d_output)
        self.v_linear = nn.Linear(d_input, d_output)
        self.k_linear = nn.Linear(d_input, d_output)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_output, d_output)

    def forward(self, q, k, v,model_name,rpos_k,rpos_v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        original_v = self.v_linear(v)
        v = original_v.view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        if "no_pe" in model_name:
            # calculate attention using original attention
            scores = attention(q, k, v, self.d_k, mask, self.dropout)
        elif "with_pe" in model_name:
            # calculate attention using attention and relative positional embedding
            scores = attention_rpos(q, k, v, rpos_k, rpos_v, self.d_k, mask, self.dropout)
        else:
            print("wrong")

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_output)
        output = self.out(concat)
        #return(output)
        return(original_v,output)

def clones(module, N): #"Produce N identical layers."
    return(nn.ModuleList([copy.deepcopy(module) for _ in range(N)]))

class EncoderLayer(nn.Module):
    def __init__(self, d_input, d_output, heads, dropout = 0.3):
        super().__init__()
        self.norm_1 = Norm(d_input)
        self.norm_2 = Norm(d_output)
        self.attn = MultiHeadAttention(heads, d_input, d_output)
        self.ff = FeedForward(d_output)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, model_name,rpos_k,rpos_v):

        #-------- syn v32 start ------
        #x = self.norm_1(x)
        #x2 = self.norm_2(self.dropout_1(self.attn(x,x,x,model_name,rpos_k,rpos_v))) # norm in v32
        #x = x+self.ff(x2)
        #-------- syn v32 end   ------

        # new version: tried for different d_input and d_output
        x = self.norm_1(x)
        original,transformed = self.attn(x,x,x,model_name,rpos_k,rpos_v)
        x = original + self.ff(self.norm_2(self.dropout_1(transformed)))
        return(x)
