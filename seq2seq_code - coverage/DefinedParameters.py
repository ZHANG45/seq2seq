# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:28:32 2018

@author: Mac
"""
import os 
#set parameters    
data = 'data/'
MODEL_FILE=data+'20190206_coverage'
BATCH_SIZE=128  
BATCH_SIZE_TEST=1 
VOCAB_SIZE=50000 
EARLY_STOP_STEP=10
MAX_LENGTH_EVALUATE=50


#changed parameters
SET='moto' #moto: train a new model.  continue : continue your training from a trained model.  evaluate: generate test result by trained model。
EVELUAGE_METHOD='greedy_decode' 
USE_COVERAGE=True

#gpu
DEVICE='cuda'
DEVICES=[2,3] #your gpu device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #always start from 0

#fixed parameters
USE_Tanh=True
USE_LengthNormalization=True

EPOCH=20
EMBED=1000
HIDDEN=1000
MAX_FERTILITY=3

#file
train_src_file = 'europarl-v7/train.en'
train_trg_file = 'europarl-v7/train.de'
valid_src_file = 'europarl-v7/newstest2013.en'
valid_trg_file = 'europarl-v7/newstest2013.de'
test_src_file = 'europarl-v7/newstest2014.en'
test_trg_file = 'europarl-v7/newstest2014.de'
TRAIN_FILE=data+'train.tsv'
VALID_FILE=data+'valid.tsv'
TEST_FILE=data+'test.tsv'
TRAINSLATION_FILE = MODEL_FILE+'_translation'+'.txt'
EVALUATE_FILE=MODEL_FILE+'_evaluate.txt'


