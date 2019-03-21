# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:30:24 2018

@author: zhang
"""
import os
import torch
from DefinedParameters import REVERSE_TRANSLATION, SET, USE_AER_THRESHOLD
def write_alignment(model, iter, file, threshold=0.3):
     i = 0
     with open(file, 'w') as f:
         for batch in iter:
             src = batch.src
             trg = batch.trg 

             #print(src,source)
             _, attn_weights = model(src, trg) #attn_weights BXlen(input)xlen(target)
             attn_weights = attn_weights.squeeze(0)
             attn_weights = attn_weights[:-1,:-1] #delete <eos>

             #write alignment
             for xi in range(0, len(attn_weights)): #xi input word, del <sos>
                 if SET == 'OptAER' or USE_AER_THRESHOLD:
                     for yi in range(0, len(attn_weights[xi])):
                         if attn_weights[xi][yi] >= threshold:
                             #f.write(str(xi)+'-'+str(yi)+' ')
                             f.write(str(i)+' '+str(xi)+' '+str(yi)+'\n')
                 else:
                     values, indexs = torch.topk(attn_weights[xi],k=1)
                     indexs = indexs.cpu().numpy()
                     for value, yi in zip(values, indexs):
                         f.write(str(i)+' '+str(xi)+' '+str(yi)+'\n')

             '''
             f.write('\n')
             if i == 5:
                 break
             '''
             i += 1
     
     print('align file has been finished')
     
from hyperopt import STATUS_OK     
def alignment_opt(args):
    print('new opt')
    model, iter, file, threshold = args
    write_alignment(model, iter, file, threshold)    
    if not REVERSE_TRANSLATION:
        gold_file = "data_ja/gold_align_file_ordered.txt"
    else:
        gold_file = "data_ja/gold_align_file_reverse.txt"
    perl_file = "word_alignment_file/mt_chinese_v1/wa_eval_align.pl"
        
    tmp = os.popen("perl " + perl_file +' '+ gold_file+ " "+file).readlines()
    aer = float(tmp[-1].split('=')[-1])
    print(aer)
    if aer - 0 < 0.1 :
        acc = 0
    else:
        acc = 1 - aer
    return {'loss': -acc, 'status': STATUS_OK }
