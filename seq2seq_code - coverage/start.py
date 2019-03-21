# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:08:19 2018

@author: zhang
"""
import torch
import torch.nn as nn
import make_iterator
from Lang import Lang
from DefinedParameters import CHOOSE_DATASET,EVELUAGE_METHOD,EMBED, HIDDEN, DEVICE, SET, MODEL_FILE, MAX_LENGTH_EVALUATE, ALIGN_FILE, AER_THRESHOLD, EVALUATE_FILE, USE_COVERAGE, EPOCH, DEVICES, EARLY_STOP_STEP
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

print(torch.device(DEVICE))
model.to(device=torch.device(DEVICE))

# Set up Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1) #Adam
lr_lambda = lambda epoch: (0.5) ** (epoch - 7) if epoch > 7 else 1 
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
criterion = nn.CrossEntropyLoss(ignore_index=1)
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
     translate_step(model, test_iter, trg_TEXT.vocab ,MAX_LENGTH_EVALUATE)
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
     load_model(model, MODEL_FILE)


# Train loop
from loss import ParallelLossCompute, LossCompute
from train_step import train_step, valid_step
min_loss =  100
early_times = 0

with open(EVALUATE_FILE,'a+') as f:
    f.write('USE_COVERAGE:'+ str(USE_COVERAGE)+'\n')
for epoch in range(1,10):
    print('epoch:', epoch, 'early_times:', early_times)
    scheduler.step()
    train_loss = train_step(train_iter, model, ParallelLossCompute(
             model.generator, criterion, DEVICES,  optimizer))
    torch.cuda.empty_cache()
    valid_loss = valid_step(valid_iter, model, ParallelLossCompute(
             model.generator, criterion, DEVICES, opt=None))

    #セーブ
    save_model(model, MODEL_FILE)
    with open(EVALUATE_FILE,'a+') as f:
        #f.write('epoch:'+str(epoch)+ ' loss train {}\t valid {}'.format(train_loss, valid_loss)+'\n')		
        f.write('epoch:'+str(epoch)+ ' loss train {}'.format(train_loss)+'\n')		
    
    print('epoch loss train {}'.format(train_loss))
    
    if valid_loss < min_loss:
        min_loss = valid_loss
        early_times = 0

    else:
        early_times += 1
	 
    if early_times == EARLY_STOP_STEP:
        print(MODEL_FILE,'early stop already satisfied, please use evaluate mode')
        break
    