# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:28:32 2018

@author: Mac
"""
import os 
#set parameters    
data = '/home/lr/zhang/multi_gpu/data/'
PAIRS_FILE=data+'20181226standford_pairs.pk'
WORD_EMBEDDING_FILE=data+'20190112_standford1000d_lang.pk' #'20181224langs_500d.pk' '20180820langs.pk'
MODEL_FILE='data/20190206model_only_coverage'
BATCH_SIZE=128  
BATCH_SIZE_TEST=1 
VOCAB_SIZE=50000 
EARLY_STOP_STEP=10
MAX_LENGTH_EVALUATE=50


#changed parameters
SET='continue' #continue, evaluate, moto, 
EVELUAGE_METHOD='greedy_decode' #beam_search greedy_decode
USE_COVERAGE=True
REVERSE_TRANSLATION=True

#gpu
DEVICE='cuda'
DEVICES=[2,3]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#fixed parameters
USE_Tanh=True
USE_LengthNormalization=True

EPOCH=20
EMBED=1000
HIDDEN=1000
MAX_FERTILITY=3

#file
train_src_file = '/home/lr/zhang/europarl-v7/train.en'
train_trg_file = '/home/lr/zhang/europarl-v7/train.de'
valid_src_file = '/home/lr/zhang/europarl-v7/newstest2013.en'
valid_trg_file = '/home/lr/zhang/europarl-v7/newstest2013.de'
test_src_file = '/home/lr/zhang/europarl-v7/newstest2014.en'
test_trg_file = '/home/lr/zhang/europarl-v7/newstest2014.de'
TRAIN_FILE=data+'train.tsv'
VALID_FILE=data+'valid.tsv'
TEST_FILE=data+'test.tsv'
TRAINSLATION_FILE = MODEL_FILE+'_translation'+'.txt'
EVALUATE_FILE=MODEL_FILE+'_evaluate.txt'

