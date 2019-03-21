# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:22:21 2018

@author: zhang
"""
import math
import torch
import torch.nn.functional as F
from DefinedParameters import TRAINSLATION_FILE, EVALUATE_FILE, EVELUAGE_METHOD, SET

def label2word(labels, vocab_itos):
    sentence = [vocab_itos[label] for label in labels]
    sentence = ' '.join(sentence)
    sentence = sentence.split('<pad>')[0]
    sentence = sentence.split('<eos>')[0]
    return sentence

# Greedy Decode
def greedy_decode(model, src, max_len, trg_vocab):
    model.eval()
    # Encoder
    encoder_states, last_states, fertility = model.encoder(src)

    # <sos>トークンをbatch_size分用意する
    sos_label = trg_vocab.stoi['<sos>']
    eos_label = trg_vocab.stoi['<eos>']
    batch_size = src[0].size(1)
    sos = torch.ones(1, batch_size).fill_(sos_label).type_as(src[0]) #1xB
    eos = torch.ones(1, batch_size).fill_(eos_label).type_as(src[0]) #1xB
    ys = sos

    for i in range(max_len-1):
        out, _ = model.decoder(
               (torch.cat([ys, eos], 0), None), encoder_states, last_states, fertility, src[0] == 1)
        prob = model.generator(out)
        _, next_word = torch.max(prob, dim = 2)
        ys = torch.cat([sos, next_word], dim=0)
    ys = ys[1:].transpose(0,1).cpu().numpy() #LxB -> BxL
    sentences = []
    for line in ys:
        sentence = label2word(line, trg_vocab.itos)
        sentences.append(sentence)
    return sentences

def translate_step(model, iter, vocab, max_len):    
     #zhangying
     with open(TRAINSLATION_FILE,'w', encoding='utf-8') as f2:
         for batch in iter:
             src = batch.src
             #else:
             #    src = batch.src
             with torch.no_grad():
                 translations = greedy_decode(model, src, max_len=max_len, trg_vocab=vocab)
             for translation in translations:
                 f2.write(translation +'\n')

     with open(EVALUATE_FILE,'a+') as f:
         f.write(SET+' '+EVELUAGE_METHOD+'\n')
         f.write('\n')