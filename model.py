import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    # input format
    # Q : tensor with size [batch_size, n, d_k]
    # K : tensor with size [batch_size, m, d_k]
    # V : tensor with size [batch_size, m, d_v]
    
    d_k = Q.size(2) # scale factor
    KT = torch.transpose(K, 1, 2) # batch_size * d_k * m
    QKT = torch.bmm(Q,KT) # batch_size * n * m
    weights = F.softmax(QKT/np.sqrt(d_k)) # batch_size * n * m
    return torch.bmm(weights, V) # batch_size * n * d_v

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(SingleHeadAttention, self).__init__()
        self.linear_q = nn.Linear(d_model, d_k)
        self.linear_k = nn.Linear(d_model, d_k)
        self.linear_v = nn.Linear(d_model, d_v)

    def forward(self, Q, K, V):
        # input format
        # Q : tensor with size [batch_size, n, d_model]
        # K : tensor with size [batch_size, m, d_model]
        # V : tensor with size [batch_size, m, d_model]

        projected_Q = self.linear_q(Q)
        projected_K = self.linear_k(K)
        projected_V = self.linear_v(V)
        Attn = scaled_dot_product_attention(projected_Q, projected_K, projected_V)
        return Attn # batch_size * n * d_v

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(MultiHeadAttention, self).__init__()
        self.heads = []
        for _ in range(h):
            self.heads.append(SingleHeadAttention(d_model, d_k, d_v))
        self.linear = nn.Linear(h*d_v, d_model)

    def forward(self, Q, K, V):
        # input format
        # Q : tensor with size [batch_size, n, d_model]
        # K : tensor with size [batch_size, m, d_model]
        # V : tensor with size [batch_size, m, d_model]

        attentions = []
        for head in self.heads:
            attentions.append(head(Q, K, V))# batch_size * n * d_v
        
        concated_attention = torch.cat(attentions, dim = 2) # batch_size * n * (h * d_v)
        return self.linear(concated_attention) # batch_size * n * d_model

class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
         

    def forward(self, x):
        pass

class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()

    def forward(self, x):
        pass

# one-hot vector encoding
class Embedding():
    def __init__(self, vocab_size):
        self.dim = vocab_size # embedding dimension

    def forward(self, x):
        # input : tensor with size [batch_size, # of words]
        batch_size = x.size(0)
        num_of_words = x.size(1)
        
        embedding = torch.zeros(batch_size, num_of_words, self.dim)
        for i in range(batch_size):
            for j in range(num_of_words):
                embedding[i][j][x[i][j]] = 1

        return embedding

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = nn.Sequential(
                        EncoderBlock(),
                        EncoderBlock(),
                        EncoderBlock(),
                        EncoderBlock(),
                        EncoderBlock(),
                        EncoderBlock())
        self.decoder = nn.Sequential(
                        DecoderBlock(),
                        DecoderBlock(),
                        DecoderBlock(),
                        DecoderBlock(),
                        DecoderBlock(),
                        DecoderBlock())
        


    def forward(self, x):
        pass