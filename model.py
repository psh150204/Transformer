import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from torch.autograd import Variable

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

class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embedding, self).__init__()
        self.dim = vocab_size # embedding dimension
        self.linear = nn.Linear(self.dim, d_model)

    def forward(self, x):
        x = torch.tensor(x)
        # input : tensor with size [batch_size, num_of_words]
        batch_size = x.size(0)
        num_of_words = x.size(1)

        # embedding
        one_hot_encoding = torch.zeros(batch_size, num_of_words, self.dim)
        for i in range(batch_size):
            for j in range(num_of_words):
                one_hot_encoding[i][j][x[i][j]] = 1

        # masking
        padding = 2
        mask = (x == padding)
        masks = []
        for elem in mask:
            masks.append(elem.unsqueeze(0).repeat(num_of_words,1).unsqueeze(0))

        input_mask = torch.cat(masks, dim = 0) # batch_size * num_of_words * num_of_words

        embedding = self.linear(one_hot_encoding) # batch_size * num_of_words * d_model
        positionally_encoded_embedding = positional_encoding(embedding)

        return positionally_encoded_embedding, input_mask


def scaled_dot_product_attention(Q, K, V, mask):
    # input format
    # Q : tensor with size [batch_size, num_of_words, d_k]
    # K : tensor with size [batch_size, num_of_words, d_k]
    # V : tensor with size [batch_size, num_of_words, d_v]
    # mask : tensor with size [batch_size, num_of_words, num_of_words]
    
    d_k = Q.size(2) # scale factor
    Kt = torch.transpose(K, -2, -1) # batch_size * d_k * num_of_words
    QKt = torch.bmm(Q,Kt) # batch_size * num_of_words * num_of_words

    if Q.size() != K.size():
        print(Q.size(), K.size())
    
    masked_QKt = QKt.masked_fill(mask == 1, -1e9)

    weights = F.softmax(masked_QKt/np.sqrt(d_k)) # batch_size * num_of_words * num_of_words
    return torch.bmm(weights, V) # batch_size * num_of_words * d_v

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(SingleHeadAttention, self).__init__()
        self.linear_q = nn.Linear(d_model, d_k)
        self.linear_k = nn.Linear(d_model, d_k)
        self.linear_v = nn.Linear(d_model, d_v)

    def forward(self, Q, K, V, mask):
        # input format
        # Q : tensor with size [batch_size, num_of_words, d_model]
        # K : tensor with size [batch_size, num_of_words, d_model]
        # V : tensor with size [batch_size, num_of_words, d_model]

        projected_Q = self.linear_q(Q)
        projected_K = self.linear_k(K)
        projected_V = self.linear_v(V)
        Attn = scaled_dot_product_attention(projected_Q, projected_K, projected_V, mask)
        return Attn # batch_size * num_of_words * d_v

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(MultiHeadAttention, self).__init__()
        self.heads = []
        for _ in range(h):
            self.heads.append(SingleHeadAttention(d_model, d_k, d_v))
        self.linear = nn.Linear(h*d_v, d_model)

    def forward(self, Q, K, V, mask):
        # input format
        # Q : tensor with size [batch_size, num_of_words, d_model]
        # K : tensor with size [batch_size, num_of_words, d_model]
        # V : tensor with size [batch_size, num_of_words, d_model]

        attentions = []
        for head in self.heads:
            attentions.append(head(Q, K, V, mask))# batch_size * num_of_words * d_v
        
        concated_attention = torch.cat(attentions, dim = 2) # batch_size * num_of_words * (h * d_v)
        return self.linear(concated_attention) # batch_size * num_of_words * d_model

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

    def forward(self, x, mask):
        # input
        # x : a tensor with size [batch_size, num_of_words, d_model]
        # mask : a tensor with size [batch_size, num_of_words, num_of_words]

        x1 = self.attention_layer(x, x, x, mask)
        x2 = x + x1 # residual sum
        x3 = self.ln1(x2)

        x4 = self.feed_forward_layer(x3)
        x5 = x3 + x4 # residual sum
        
        return self.ln2(x5)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, h):
        super(DecoderBlock, self).__init__()
        self.self_attention_layer = MultiHeadAttention(d_model, d_k, d_v, h)
        self.enc_dec_attention_layer = MultiHeadAttention(d_model, d_k, d_v, h)
        self.feed_forward_layer = nn.Sequential(
                                    nn.Linear(d_model, d_ff),
                                    nn.ReLU(),
                                    nn.Linear(d_ff, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, trg_mask, K, V, src_mask):
        # input
        # x : a tensor with size [batch_size, num_of_words, d_model]
        # K : a tensor with size [batch_size, num_of_words, d_model]
        # V : a tensor with size [batch_size, num_of_words, d_model]

        num_of_words = x.size(1)
        
        self_mask = np.triu(np.ones((1, num_of_words, num_of_words)), k=1)
        self_mask = Variable(torch.from_numpy(self_mask) == 1) # batch_size * num_of_words * num_of_words
        
        x1 = self.self_attention_layer(x, x, x, trg_mask | self_mask)
        x2 = x + x1 # residual path
        x3 = self.ln1(x2)

        x4 = self.enc_dec_attention_layer(x3, K, V, src_mask)
        x5 = x3 + x4 # residual path
        x6 = self.ln2(x5)

        x7 = self.feed_forward_layer(x6)
        x8 = x6 + x7 # residual path
        return self.ln3(x8)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Transformer(nn.Module):
    def __init__(self, num_enc, num_dec, src_vocab_size, trg_vocab_size, d_model, d_k, d_v, d_ff, h):
        super(Transformer, self).__init__()
        self.num_enc = num_enc
        self.num_dec = num_dec
        self.src_embedding = Embedding(d_model, src_vocab_size)
        self.trg_embedding = Embedding(d_model, trg_vocab_size)
        self.encoder_layers = get_clones(EncoderBlock(d_model, d_k, d_v, d_ff, h), num_enc)
        self.decoder_layers = get_clones(DecoderBlock(d_model, d_k, d_v, d_ff, h), num_dec)
        self.linear = nn.Linear(d_model, trg_vocab_size)
        
    def forward(self, src, trg):
        src, src_mask = self.src_embedding(src)
        for i in range(self.num_enc):
            src = self.encoder_layers[i](src, src_mask)
        
        trg, trg_mask = self.trg_embedding(trg)
        for i in range(self.num_dec):
            trg = self.decoder_layers[i](trg, trg_mask, src, src, src_mask)
        
        return self.linear(trg) # batch_size * num_of_words * trg_vocab_size