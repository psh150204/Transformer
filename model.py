import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def positional_encoding(x):
    # input : embedded vector (a tensor with size [batch_size, num_of_words, d_model])
    batch_size = x.size(0)
    num_of_words = x.size(1)
    d_model = x.size(2)
    
    pe_base = torch.zeros(1, num_of_words, d_model)
    for pos in range(num_of_words):
        for i in range(d_model):
            if i % 2 == 0:
                pe_base[0][pos][i] = np.cos((pos + 1) / (10000 ** (float(i) / d_model)))
            else:
                pe_base[0][pos][i] = np.sin((pos + 1) / (10000 ** (float(i + 1) / d_model)))
    
    pe = pe_base.repeat(batch_size, 1, 1) # batch_size * num_of_words * d_model

    return pe + x

def scaled_dot_product_attention(Q, K, V, mask):
    # input format
    # Q : tensor with size [batch_size, n, d_k]
    # K : tensor with size [batch_size, m, d_k]
    # V : tensor with size [batch_size, m, d_v]
    
    d_k = Q.size(2) # scale factor
    Kt = torch.transpose(K, -2, -1) # batch_size * d_k * m
    QKt = torch.bmm(Q,Kt) # batch_size * n * m
    masked_QKt = torch.bmm(QKt, mask)
    weights = F.softmax(masked_QKt/np.sqrt(d_k)) # batch_size * n * m
    return torch.bmm(weights, V) # batch_size * n * d_v

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(SingleHeadAttention, self).__init__()
        self.linear_q = nn.Linear(d_model, d_k)
        self.linear_k = nn.Linear(d_model, d_k)
        self.linear_v = nn.Linear(d_model, d_v)

    def forward(self, Q, K, V, mask):
        # input format
        # Q : tensor with size [batch_size, n, d_model]
        # K : tensor with size [batch_size, m, d_model]
        # V : tensor with size [batch_size, m, d_model]

        projected_Q = self.linear_q(Q)
        projected_K = self.linear_k(K)
        projected_V = self.linear_v(V)
        Attn = scaled_dot_product_attention(projected_Q, projected_K, projected_V, mask)
        return Attn # batch_size * n * d_v

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(MultiHeadAttention, self).__init__()
        self.heads = []
        for _ in range(h):
            self.heads.append(SingleHeadAttention(d_model, d_k, d_v))
        self.linear = nn.Linear(h*d_v, d_model)

    def forward(self, Q, K, V, mask):
        # input format
        # Q : tensor with size [batch_size, n, d_model]
        # K : tensor with size [batch_size, m, d_model]
        # V : tensor with size [batch_size, m, d_model]

        attentions = []
        for head in self.heads:
            attentions.append(head(Q, K, V, mask))# batch_size * n * d_v
        
        concated_attention = torch.cat(attentions, dim = 2) # batch_size * n * (h * d_v)
        return self.linear(concated_attention) # batch_size * n * d_model

class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, h):
        super(EncoderBlock, self).__init__()
        self.attention_layer = MultiHeadAttention(d_model, d_k, d_v, h)
        self.feed_forward_layer = nn.Sequential(
                                    nn.Linear(d_model, d_ff),
                                    nn.ReLU(),
                                    nn.Linear(d_ff, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # input : a tensor with size [batch_size, num_of_words, d_model]
        mask = 

        x1 = self.attention_layer(x, x, x, mask)
        x2 = x + x1 # residual sum
        x3 = self.ln1(x2)

        x4 = self.feed_forward_layer(x3)
        x5 = x3 + x4 # residual sum
        
        return self.ln2(x5)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(DecoderBlock, self).__init__()
        self.self_attnetion_layer = MultiHeadAttention(d_model, d_k, d_v, h)
        self.enc_dec_attention_layer = MultiHeadAttention(d_model, d_k, d_v, h)
        self.feed_forward_layer = nn.Sequential(
                                    nn.Linear(d_model, d_ff),
                                    nn.ReLU(),
                                    nn.Linear(d_ff, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, K, V):
        # input : a tensor with size [batch_size, num_of_words, d_model]
        mask1 = 
        mask2 = 
        x1 = self.self_attention_layer(x, x, x, mask1)
        x2 = x + x1 # residual path
        x3 = nn.ln1(x2)

        x4 = self.enc_dec_attention_layer(x3, K, V, mask2)
        x5 = x3 + x4 # residual path
        x6 = nn.ln2(x5)

        x7 = self.feed_forward_layer(x6)
        x8 = x6 + x7 # residual path
        return self.ln3(x8)

# one-hot vector encoding
class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        self.dim = vocab_size # embedding dimension
        self.linear = nn.Linear(self.dim, d_model)

    def forward(self, x):
        # input : tensor with size [batch_size, # of words]
        batch_size = x.size(0)
        num_of_words = x.size(1)
        
        embedding = torch.zeros(batch_size, num_of_words, self.dim)
        for i in range(batch_size):
            for j in range(num_of_words):
                embedding[i][j][x[i][j]] = 1

        return self.linear(embedding) # batch_size * num_of_words * d_model

class Transformer(nn.Module):
    def __init__(self, num_enc, num_dec):
        super(Transformer, self).__init__()
        self.encoder = []
        for _ in range(num_enc):
            self.encoder.append(EncoderBlock())
        self.decoder = []
        for _ in range(num_dec):
            self.decoder.append(DecoderBlock())
        
    def forward(self, x):
        pass