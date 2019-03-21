# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:17:09 2018

@author: zhang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from DefinedParameters import USE_COVERAGE, HIDDEN

class Attention(nn.Module):
     def __init__(self, method):
          super(Attention, self).__init__()

          self.method = method
          self.hidden_size = HIDDEN

          if self.method == 'general':
             self.attn = nn.Linear(self.hidden_size, self.hidden_size)
          elif self.method == 'concat':
             self.matrix_W = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
             self.matrix_U = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
             self.alignment_layer =  nn.Linear(self.hidden_size, 1, bias=False)
             if USE_COVERAGE:
                 self.matrix_V = nn.Linear(1, self.hidden_size, bias=False)


     def forward(self, decoder_state, encoder_states, attn_weights, src_mask, trg_mask, fertility=0):
          #fertility #BxLen(input)
          #encoder_states LxBxH
          #decoder_state  1xBxH
          #attn_weights BXlen(input)xlen(target words has been delt)
          #src_mask  Len(input)xB
          #trg_mask  len(target)xB


          encoder_states = encoder_states.transpose(0, 1) #BxLxH
          # 内積でスコアを計算
          if self.method == 'dot':
              decoder_state = decoder_state.transpose(0,1).transpose(1,2) # BXHX1
              score = torch.bmm(encoder_states, decoder_state) #BxLx1
          elif self.method == 'concat':

              batch_size, input_length, hidden_size = encoder_states.size() 
              h_t = decoder_state.transpose(0, 1) 
              #h_t = h_t[-1].unsqueeze(0).transpose(0, 1) #Bx1xH
              h_t = h_t.expand(batch_size, input_length, hidden_size) #BxLxH
              
              if USE_COVERAGE:
                  if attn_weights.size(2) != trg_mask.size(0):
                      zero = torch.zeros(attn_weights.size(0), attn_weights.size(1), trg_mask.size(0)-attn_weights.size(2)).cuda()
                      attn_weights = torch.cat((attn_weights, zero),2) #BXlen(input)xlen(target)
                  attn_weights = attn_weights.transpose(0,1).unsqueeze(3).transpose(2,3) #len(input)xBx1xlen(target)
                  trg_mask = trg_mask.transpose(0, 1).unsqueeze(2).float()  #Bxlen(target)x1
                  #print(attn_weights.size(), trg_mask.size())
                  attn_weights = torch.matmul(attn_weights, trg_mask).squeeze(2).squeeze(2).transpose(0,1) #Bxlen(input)	  
                  coverage = torch.div(attn_weights, fertility).unsqueeze(2) #BXLX1
              
                  score = self.alignment_layer(torch.tanh(self.matrix_W(h_t) + self.matrix_U(encoder_states) + self.matrix_V(coverage)))
              else:
                  score = self.alignment_layer(torch.tanh(self.matrix_W(h_t) + self.matrix_U(encoder_states)))

          elif self.method == 'general':
              decoder_state = decoder_state.transpose(0,1).transpose(1,2) # BXHX1
              score = self.attn(encoder_states)  #BXLXH
              score = torch.bmm(score, decoder_state)
          # Paddingに当たる部分の値を-700へ(softmaxで0に落ちるように)
          src_mask = src_mask.transpose(0, 1).unsqueeze(2) #BxLx1  (0,0,0,1,1,...,1)
          score = torch.where(src_mask, torch.full_like(score, -700), score)

          # softmaxで確率化
          weight = F.softmax(score, dim=1) # BxLx1
          context = torch.matmul(weight.transpose(1,2), encoder_states).squeeze(1)
          return context, weight