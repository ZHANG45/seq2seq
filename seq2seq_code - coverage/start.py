# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:08:19 2018

@author: zhang
"""
import torch
import torch.nn as nn
import make_iterator
from Lang import Lang
from DefinedParameters import USE_PRETRAINED_EMBEDDING,REVERSE_TRANSLATION,CHOOSE_DATASET,EVELUAGE_METHOD,EMBED, HIDDEN, DEVICE, SET, MODEL_FILE, MAX_LENGTH_EVALUATE, ALIGN_FILE, AER_THRESHOLD, EVALUATE_FILE, WITH_A, USE_ZY, USE_COVERAGE, EPOCH, PARALLEL, DEVICES, EARLY_STOP_STEP
#check settings
print('file:',MODEL_FILE)
print('set:',SET, EVELUAGE_METHOD)
print('dataset:',CHOOSE_DATASET)
print(torch.device(DEVICE), DEVICES)
print('WITH_A:', WITH_A)
print('USE_ZY:', USE_ZY)
print('USE_COVERAGE:',USE_COVERAGE)
print('REVERSE_TRANSLATION:', REVERSE_TRANSLATION)
print('USE_PRETRAINED_EMBEDDING:',USE_PRETRAINED_EMBEDDING)
#start
# Set up Model
from model import Seq2Seq
train_iter, valid_iter, test_iter, src_TEXT, ml_TEXT, trg_TEXT = make_iterator.MAKE_ITER()
model = Seq2Seq(len(src_TEXT.vocab.itos),len(ml_TEXT.vocab.itos), len(trg_TEXT.vocab.itos), EMBED, HIDDEN, 0.2, src_TEXT.vocab.vectors, ml_TEXT.vocab.vectors, trg_TEXT.vocab.vectors)

print(torch.device(DEVICE))
model.to(device=torch.device(DEVICE))

# Set up Optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001) #Adam
criterion = nn.NLLLoss(ignore_index=1)
criterion_cos = nn.CosineEmbeddingLoss()

def save_model(model, file_name):
     torch.save(model.state_dict(), file_name)
     print('model has been saved')
     
def load_model(model, file_name):
     model.load_state_dict(torch.load(file_name)) 
     print('model has been loaded')

"""手元にモデルをダウンロードすることも可能"""


"""## モデルの学習

モデルを学習する  
毎エポックでtrain_stepとvalid_stepを繰り返す。
"""
from translate import translate_step
from alignment import write_alignment, alignment_opt
if SET == 'evaluate':
     load_model(model, MODEL_FILE)
     test_src_en= "data/src.test.sorted"
     src_en_lines = []
     with open(test_src_en, 'r') as f_en:
         for ind, line in enumerate(f_en):
             src_en_lines.append(line.strip().split())
     translate_step(model, src_en_lines, test_iter, src_TEXT.vocab , trg_TEXT.vocab , MAX_LENGTH_EVALUATE)
     exit()

elif SET == 'alignment':
     load_model(model, MODEL_FILE)
     write_alignment(model, test_iter, ALIGN_FILE, AER_THRESHOLD)
     exit()

elif SET == 'OptAER':
     from hyperopt import fmin, tpe, hp
     load_model(model, MODEL_FILE)
     
     OptimizeTime = 20
     space = [model, test_iter, ALIGN_FILE, hp.uniform('threshold', 0, 1)]
    
     #space=hp.uniform('x', -10, 10)
     best = fmin(alignment_opt,
                 space,
                 algo=tpe.suggest,
                 max_evals=OptimizeTime)
     USE_AER_THRESHOLD=True
     print(best['threshold'])
     write_alignment(model, test_iter, ALIGN_FILE, best['threshold'])
     exit()
     
elif SET == 'continue':
     load_model(model, MODEL_FILE) #MODEL_FILE


# Train loop
from loss import ParallelLossCompute, LossCompute
from train_step import train_step, valid_step
min_loss =  100
early_times = 0
if USE_ZY:
    #vi_embedding_matrix= model.encoder.embed(torch.LongTensor(range(len(src_TEXT.vocab.itos))).cuda()).unsqueeze(2) #Vocab_sizexEmbed_sizex1
    vi_embedding_matrix = len(src_TEXT.vocab.itos)
else:
    vi_embedding_matrix = None
with open(EVALUATE_FILE,'a+') as f:
    f.write('WITH_A:'+ str(WITH_A)+'\n')
    f.write('USE_ZY:'+ str(USE_ZY)+'\n')
    f.write('USE_COVERAGE:'+ str(USE_COVERAGE)+'\n')
    #f.write('REVERSE_TRANSLATION:'+str( REVERSE_TRANSLATION)+'\n')
    f.write('USE_PRETRAINED_EMBEDDING:'+str(USE_PRETRAINED_EMBEDDING) + '\n')
for epoch in range(9,13):
    print('epoch:', epoch, 'early_times:', early_times)
    if PARALLEL:
        train_loss = train_step(train_iter, model, ParallelLossCompute(
                 model.generator, model.generator_q, criterion, criterion_cos, DEVICES, USE_ZY, vi_embedding_matrix, optimizer))
        valid_loss = valid_step(valid_iter, model, ParallelLossCompute(
                 model.generator, model.generator_q, criterion, criterion_cos, DEVICES, USE_ZY, vi_embedding_matrix, opt=None))
    else:
        train_loss = train_step(train_iter, model, #model.generator_q
                                 LossCompute(model.generator, model.generator_q, criterion, criterion_cos, USE_ZY, vi_embedding_matrix, optimizer))
        valid_loss = valid_step(valid_iter, model,
                                 LossCompute(model.generator, model.generator_q, criterion, criterion_cos, USE_ZY, vi_embedding_matrix, opt=None))
    #セーブ
    save_model(model, MODEL_FILE)
    with open(EVALUATE_FILE,'a+') as f:
        #f.write('epoch:'+str(epoch)+ ' loss train {}\t valid {}'.format(train_loss, valid_loss)+'\n')		
        f.write('epoch:'+str(epoch)+ ' loss train {} , valid {}'.format(train_loss, valid_loss)+'\n')		
    
    print('epoch loss train {} , valid {}'.format(train_loss, valid_loss))
    
    if valid_loss < min_loss:
        min_loss = valid_loss
        early_times = 0

    else:
        early_times += 1
	 
    if early_times == EARLY_STOP_STEP:
        print(MODEL_FILE,'early stop already satisfied, please use evaluate mode')
        break
    