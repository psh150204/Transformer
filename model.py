import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from torch.autograd import Variable

def positional_encoding(x):
    # input : embedded vector (a tensor with size [batch_size, sentence_length, d_model])
    batch_size = x.size(0)
    sentence_length = x.size(1)
    d_model = x.size(2)
    
    pe_base = Variable(torch.zeros(1, sentence_length, d_model)).cuda()
    for pos in range(sentence_length):
        for i in range(d_model):
            if i % 2 == 0:
                pe_base[0][pos][i] = np.cos((pos + 1) / (10000 ** (float(i) / d_model)))
            else:
                pe_base[0][pos][i] = np.sin((pos + 1) / (10000 ** (float(i + 1) / d_model)))
    
    pe = pe_base.repeat(batch_size, 1, 1) # batch_size * sentence_length * d_model

    return pe + x

class OneHotVectorEncoding(nn.Module):
    def __init__(self, vocab_size):
        super(OneHotVectorEncoding, self).__init__()
        self.dim = vocab_size

    def forward(self, x):
        # input : tensor with size [batch_size, sentence_length]
        batch_size = x.size(0)
        sentence_length = x.size(1)

        # embedding
        one_hot_encoding = Variable(torch.zeros(batch_size, sentence_length, self.dim)).cuda()
        for i in range(batch_size):
            for j in range(sentence_length):
                one_hot_encoding[i][j][x[i][j]] = 1

        return one_hot_encoding

class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embedding, self).__init__()
        self.embedder = OneHotVectorEncoding(vocab_size)
        self.linear = nn.Linear(vocab_size, d_model)

    def forward(self, x):
        one_hot_encoding = self.embedder(x)
        embedding = self.linear(one_hot_encoding) # batch_size * sentence_length * d_model
        positionally_encoded_embedding = positional_encoding(embedding)

        return positionally_encoded_embedding


def scaled_dot_product_attention(Q, K, V, mask, dropout):
    # input format
    # Q : tensor with size [batch_size, sentence_length1, d_k]
    # K : tensor with size [batch_size, sentence_length2, d_k]
    # V : tensor with size [batch_size, sentence_length2, d_v]
    # mask : tensor with size [batch_size, sentence_length1, sentence_length2]
    
    d_k = Q.size(-1) # scale factor
    Kt = torch.transpose(K, -2, -1) # batch_size * d_k * sentence_length2
    QKt = torch.bmm(Q,Kt) # batch_size * sentence_length1 * sentence_length2
    
    masked_QKt = QKt.masked_fill(mask == 1, -1e9)

    weights = F.softmax(masked_QKt/np.sqrt(d_k), dim = -1) # batch_size * sentence_length1 * sentence_length2
    weights = dropout(weights)
    return torch.bmm(weights, V) # batch_size * sentence_length1 * d_v

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(SingleHeadAttention, self).__init__()
        self.linear_q = nn.Linear(d_model, d_k)
        self.linear_k = nn.Linear(d_model, d_k)
        self.linear_v = nn.Linear(d_model, d_v)

    def forward(self, Q, K, V, mask, dropout):
        # input format
        # Q : tensor with size [batch_size, sentence_length, d_model]
        # K : tensor with size [batch_size, sentence_length, d_model]
        # V : tensor with size [batch_size, sentence_length, d_model]

        projected_Q = self.linear_q(Q)
        projected_K = self.linear_k(K)
        projected_V = self.linear_v(V)
        Attn = scaled_dot_product_attention(projected_Q, projected_K, projected_V, mask, dropout)
        return Attn # batch_size * sentence_length * d_v

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, p_drop = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.heads = get_clones(SingleHeadAttention(d_model, d_k, d_v), h)
        self.dropout = nn.Dropout(p_drop)
        self.linear = nn.Linear(h*d_v, d_model)

    def forward(self, Q, K, V, mask):
        # input format
        # Q : tensor with size [batch_size, sentence_length, d_model]
        # K : tensor with size [batch_size, sentence_length, d_model]
        # V : tensor with size [batch_size, sentence_length, d_model]

        attentions = []
        for i in range(self.h):
            attentions.append(self.heads[i](Q, K, V, mask, self.dropout))# batch_size * sentence_length * d_v
        
        concated_attention = torch.cat(attentions, dim = 2) # batch_size * sentence_length * (h * d_v)
        return self.linear(concated_attention) # batch_size * sentence_length * d_model

class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, h, p_drop = 0.1):
        super(EncoderBlock, self).__init__()
        self.attention_layer = MultiHeadAttention(d_model, d_k, d_v, h)
        self.feed_forward_layer = nn.Sequential(
                                    nn.Linear(d_model, d_ff),
                                    nn.ReLU(),
                                    nn.Dropout(p_drop),
                                    nn.Linear(d_ff, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)
        self.dropout2 = nn.Dropout(p_drop)

    def forward(self, x, mask):
        # input
        # x : a tensor with size [batch_size, sentence_length, d_model]
        # mask : a tensor with size [batch_size, sentence_length, sentence_length]
        x1 = self.attention_layer(x, x, x, mask)
        x2 = x + self.dropout1(x1) # residual sum
        x3 = self.ln1(x2)

        x4 = self.feed_forward_layer(x3)
        x5 = x3 + self.dropout2(x4) # residual sum
        
        return self.ln2(x5)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, h, p_drop = 0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention_layer = MultiHeadAttention(d_model, d_k, d_v, h)
        self.enc_dec_attention_layer = MultiHeadAttention(d_model, d_k, d_v, h)
        self.feed_forward_layer = nn.Sequential(
                                    nn.Linear(d_model, d_ff),
                                    nn.ReLU(),
                                    nn.Dropout(p_drop),
                                    nn.Linear(d_ff, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)
        self.dropout2 = nn.Dropout(p_drop)
        self.dropout3 = nn.Dropout(p_drop)

    def forward(self, x, trg_mask, K, V, memory_mask):
        # input
        # x : a tensor with size [batch_size, sentence_length, d_model]
        # K : a tensor with size [batch_size, sentence_length, d_model]
        # V : a tensor with size [batch_size, sentence_length, d_model]

        sentence_length = x.size(1)
        
        self_mask = np.triu(np.ones((1, sentence_length, sentence_length)), k=1)
        self_mask = Variable(torch.from_numpy(self_mask) == 1).cuda() # batch_size * sentence_length * sentence_length
        
        x1 = self.self_attention_layer(x, x, x, trg_mask | self_mask)
        x2 = x + self.dropout1(x1) # residual path
        x3 = self.ln1(x2)

        x4 = self.enc_dec_attention_layer(x3, K, V, memory_mask)
        x5 = x3 + self.dropout2(x4) # residual path
        x6 = self.ln2(x5)

        x7 = self.feed_forward_layer(x6)
        x8 = x6 + self.dropout3(x7) # residual path
        return self.ln3(x8)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def generate_masks(src, trg, src_padding, trg_padding):
    # input
    # src : a tensor with size [batch_size, src_sentence_length]
    # trg : a tensor with size [batch_size, trg_sentence_length]

    src_n = src.size(1) # src_sentence_length
    trg_n = trg.size(1) # trg_sentence_length

    src_padding_mask = (src == src_padding).unsqueeze(1) # batch_size * 1 * src_sentence_length
    trg_padding_mask = (trg == trg_padding).unsqueeze(1) # batch_size * 1 * trg_sentence_length

    src_mask = src_padding_mask.repeat(1, src_n, 1) # batch_size * src_sentence_length * src_sentence_length
    memory_mask = src_padding_mask.repeat(1, trg_n, 1) # batch_size * trg_sentence_length * src_sentence_length
    trg_mask = trg_padding_mask.repeat(1, trg_n, 1) # batch_size * trg_sentence_length * trg_sentence_length

    return src_mask, memory_mask, trg_mask

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
        src_mask, memory_mask, trg_mask = generate_masks(src, trg, 2, 2)

        src = self.src_embedding(src)
        for i in range(self.num_enc):
            src = self.encoder_layers[i](src, src_mask)
        
        trg = self.trg_embedding(trg)
        for i in range(self.num_dec):
            trg = self.decoder_layers[i](trg, trg_mask, src, src, memory_mask)
        
        return self.linear(trg) # batch_size * sentence_length * trg_vocab_size