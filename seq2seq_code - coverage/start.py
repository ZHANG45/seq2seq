# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:08:19 2018

@author: zhang
"""
import torch
import torch.nn as nn
import make_iterator
from DefinedParameters import CHOOSE_DATASET,EVELUAGE_METHOD,EMBED, HIDDEN, DEVICE, SET, MODEL_FILE, MAX_LENGTH_EVALUATE, EVALUATE_FILE, USE_COVERAGE, EPOCH, DEVICES, EARLY_STOP_STEP
#check settings
print('file:',MODEL_FILE)
print('set:',SET, EVELUAGE_METHOD)
print('dataset:',CHOOSE_DATASET)
print(torch.device(DEVICE), DEVICES)
print('USE_COVERAGE:',USE_COVERAGE)

#start

# Set up Model
from model import Seq2Seq
train_iter, valid_iter, test_iter, src_TEXT, trg_TEXT = make_iterator.MAKE_ITER()
model = Seq2Seq(len(src_TEXT.vocab.itos), len(trg_TEXT.vocab.itos), EMBED, HIDDEN, 0.2, src_TEXT.vocab.vectors,trg_TEXT.vocab.vectors)
model.to(device=torch.device(DEVICE))

# print log
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]
print(repr(model) + "\n\n")
print('total number of parameters: %d\n\n' % param_count)

# Set up Optimizer
optimizer = torch.optim.Adam(model.parameters())
#optimizer = torch.optim.SGD(model.parameters(), lr=1) #Adam
#lr_lambda = lambda epoch: (0.5) ** (epoch - 7) if epoch > 7 else 1 
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
criterion = nn.CrossEntropyLoss(ignore_index=1)

def save_model(model, file_name, epoch):
    checkpoints = {
        'epoch': epoch,        
        'model': model.state_dict()
    }
    torch.save(checkpoints, file_name)
    print('model has been saved')
     
def load_model(model, file_name):
    model.load_state_dict(torch.load(file_name)['model']) 
    print('model(epoch %s) has been loaded' % torch.load(file_name)['epoch'])

def make_save(pastscores, score):
    if max(pastscores) > float(score):
        idx = pastscores.index(max(pastscores))
        pastscores[idx] = float(score)
    else:
        idx = 3

    return idx+1 if idx <3 else False

from translate import translate_step
if SET == 'evaluate':
     load_model(model, MODEL_FILE)
     translate_step(model, test_iter, trg_TEXT.vocab ,MAX_LENGTH_EVALUATE)
     exit()

elif SET == 'continue':
     load_model(model, MODEL_FILE)

# Train loop
from loss import ParallelLossCompute
from train_step import train_step, valid_step
min_loss =  100
early_times = 0
scores = [100000, 100000, 100000]

with open(EVALUATE_FILE,'a+') as f:
    f.write('USE_COVERAGE:'+ str(USE_COVERAGE)+'\n')
    
for epoch in range(1,EPOCH+1):
    print('epoch:', epoch, 'early_times:', early_times)
    #scheduler.step()
    try:
        train_loss = train_step(train_iter, model, ParallelLossCompute(
                 model.generator, criterion, DEVICES,  optimizer))
        valid_loss = valid_step(valid_iter, model, ParallelLossCompute(
                 model.generator, criterion, DEVICES, opt=None))

    except RuntimeError as e:
        if 'out of memory' in str(e):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise e

    #セーブ
    with open(EVALUATE_FILE,'a+') as f:
        #f.write('epoch:'+str(epoch)+ ' loss train {}\t valid {}'.format(train_loss, valid_loss)+'\n')		
        f.write('epoch:'+str(epoch)+ ' loss train {}'.format(train_loss)+'\n')		
    print('epoch loss train {}'.format(train_loss))
    
    save_no = make_save(scores, valid_loss, epoch)
    
    if save_no:
        save_model(model, MODEL_FILE+'_best'+str(save_no), epoch)

    if valid_loss < min_loss:
        min_loss = valid_loss
        early_times = 0

    else:
        early_times += 1
	 
    if early_times == EARLY_STOP_STEP:
        print(MODEL_FILE,'early stop already satisfied, please use evaluate mode')
        break
    