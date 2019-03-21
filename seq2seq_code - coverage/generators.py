# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:18:26 2018

@author: zhang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_p):
        super(Generator, self).__init__()
        self.Wo = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, decoder_states):
        if type(decoder_states) is list:
            decoder_states = torch.stack(decoder_states, 0) #LxBxH
        # 語彙サイズのベクトルへと写像
        decoder_states = self.dropout(decoder_states)
        out = self.Wo(decoder_states)
        return out
    