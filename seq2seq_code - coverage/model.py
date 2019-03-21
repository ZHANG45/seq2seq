# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:19:35 2018

@author: zhang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

from generators import Generator
from attention import Attention
from DefinedParameters import  USE_COVERAGE, MAX_FERTILITY

def _inverse_indices(indices):
    indices = indices.cpu().numpy()
    r = numpy.empty_like(indices)
    r[indices] = numpy.arange(len(indices))
    return r

# Define Seq2Seq with Attention Model
class Seq2Seq(nn.Module):
    def __init__(self,
                 src_vocab_size, trg_vocab_size,
                 embed_size, hidden_size, dropout_p, src_emb_vectors, trg_emb_vectors):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size, dropout_p, src_emb_vectors)
        self.decoder = Decoder(trg_vocab_size, embed_size, hidden_size, dropout_p, trg_emb_vectors)
        self.generator = Generator(trg_vocab_size, hidden_size, dropout_p)

    def forward(self, src, trg):
        if len(src[1].size()) == 2:
            # for compatibility of parallel
            src = (src[0], src[1].squeeze(0))

        return self.decoder(trg, *self.encoder(src), src[0] == 1)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout_p, emb_vectors):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = 4
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=self.n_layers,
                            dropout=dropout_p, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout_p)
        if USE_COVERAGE:
            self.matrix_Uf = nn.Linear(2*hidden_size, 1) #compute fertility

    def forward(self, src):
        # 単語と文長に分解
        words = src[0] #LxB
        lengths = src[1] #B
        total_length=words.size()[0]

        # 入力を文長で降順ソート
        lengths, parm_idx = torch.sort(lengths, 0, descending=True)
        device = parm_idx.device
        parm_idx_rev = torch.tensor(_inverse_indices(parm_idx), device=device)
        words = words[:, parm_idx]

        # Embedding
        ex = self.embed(words) #LxB -> LxBxH
        ex = self.dropout(ex)

        # 入力をLSTMで処理するためにPackする
        packed_input = nn.utils.rnn.pack_padded_sequence(ex, lengths)

        # LSTM (L 回処理される)
        packed_output, (ht, ct) = self.lstm(packed_input)

        # LSTMの出力をUnPackしてPaddingする
        output = nn.utils.rnn.pad_packed_sequence(packed_output, total_length=total_length)

        # ソートを元の順へ戻す
        output = output[0][:, parm_idx_rev]  #LxBx2H
        ht = ht[:,parm_idx_rev]
        ct = ct[:,parm_idx_rev]
        if USE_COVERAGE:
            self.fertility = self.compute_fertility(output)
        else:
            self.fertility = None
        # 前向きと後ろ向きの平均を計算
        L, B, _ = output.size()
        output = output.view(L, B, 2, -1).sum(dim=2)
        #n_layers*2xBxH -> n_layersxBxH
        ht = torch.mean(ht.view(self.n_layers, 2, -1, self.hidden_size), 1)
        ct = torch.mean(ct.view(self.n_layers, 2, -1, self.hidden_size), 1)

        return output, (ht, ct),  self.fertility	
    
    def compute_fertility(self, hidden):
        L, B, H = hidden.size()
        #compute fertility

        Uh = self.matrix_Uf(hidden.transpose(0,1)) #BXLX1
        fertility = F.logsigmoid(Uh.transpose(1,2).transpose(0,1)).squeeze(0) * MAX_FERTILITY #BxL
        return fertility
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout_p, emb_vectors):
        super(Decoder, self).__init__()
          
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.n_layers = 4
        self.lstm = nn.LSTM(self.embed_size+self.hidden_size, hidden_size,
                            num_layers=self.n_layers, dropout=dropout_p)          
          
        self.Wc = nn.Linear(2*hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_p)
        if USE_COVERAGE:
            self.attention = Attention('concat')		  
        else:
            self.attention = Attention('general')		  

    def forward(self, trg, encoder_states, last_states, fertility, src_mask):
        words    = trg[0][:-1] #LxB (Lは<eos>を抜いた長さ)
        trg_mask = trg[0] != 1 #Len(target)xB
        #encoder_states # LxBxH
        #src_mask #LxB
        #last_states[0], [1] #n_layersxBxH
        context = None
        output = []
        self.fertility = fertility
        attn_weights = torch.zeros(encoder_states.size()[1],encoder_states.size(0),1) #BXlen(input)xlen(target)
        if torch.cuda.is_available():
            attn_weights = attn_weights.cuda() 
        for word in words: #一単語ずつ処理する teacher forcing		  
            #word Bx1
            #weight Bxlen(input)x1
            out, context, weight, last_state = self.forward_step(
                     word, context, encoder_states, last_states, attn_weights, src_mask, trg_mask)
            output.append(out)
            attn_weights = torch.cat((attn_weights,weight), 2)    
        attn_weights = attn_weights[:, :, 1:] #BXlen(input)xlen(target)
        return output

    def forward_step(self, word, context, encoder_states, states, attn_weights, src_mask, trg_mask):
        # Embedding
        ex = self.embed(word) #BxH
        ex = self.dropout(ex)

        # input-feed
        if context is None:
            context = torch.zeros(ex.size()[0], self.hidden_size).type_as(ex)
        rnn_input = torch.cat((ex, context), 1) #Bx2H
        # LSTMの入力に文長の次元がいるため拡張
        rnn_input = torch.unsqueeze(rnn_input, 0) #1xBx2H
        # LSTM(1単語ずつ)

        rnn_output, status = self.lstm(rnn_input, states)
        # 文脈ベクトルを計算
        context, weight = self.attention(rnn_output, encoder_states, attn_weights, src_mask, trg_mask, self.fertility) 
        # 拡張した次元を元に戻す
        rnn_output = torch.squeeze(rnn_output, 0)
        t = torch.cat((rnn_output, context), 1) #Bx2H
        # 次元を元に戻す(2H->H)
        t = self.dropout(t)
        out = self.Wc(t) #BxH
        return out, context, weight,status