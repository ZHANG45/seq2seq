# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:28:32 2018

@author: Mac
"""
import os 
#set parameters
USE_DE2EN_DATA=True #else use ja2en
USE_SMALL=True
if not USE_DE2EN_DATA:
    USE_SMALL=False
    data='data_ja/'
    SMALL = ''
    symbol = ''
    PAIRS_FILE=data+'20181009pairs_all_u3000.pk'
    WORD_EMBEDDING_FILE=data+'20181009langs_all_u3000.pk'
    MODEL_FILE=data+'20181220model_koba_ba_lstm_coverage_noa_reverse'    
    TUNE_FILE=data+SMALL+'tune.tsv'    
    BATCH_SIZE=80 #200 for baseline train , 180 for zy_witha, 140, 100
    BATCH_SIZE_TEST=1 #100 for test, 50
    MIN_FREQ=2
    EARLY_STOP_STEP=15
    MAX_LENGTH=40
    MAX_LENGTH_EVALUATE=80 #60 for greedy
    
else:
    data = 'data/'
    if USE_SMALL:
        SMALL = 'small'
        MAX_LENGTH=50
        symbol = '_'
    else:
        SMALL = ''
        MAX_LENGTH=50
        symbol = ''
    
    PAIRS_FILE=data+'20181226standford_pairs.pk'
    WORD_EMBEDDING_FILE=data+'20190112_standford1000d_lang.pk' #'20181224langs_500d.pk' '20180820langs.pk'
    MODEL_FILE=data + '20190206model_only_coverage_reverse'+symbol+SMALL
    TUNE_FILE=None
    BATCH_SIZE=128 #200 for baseline train , 180 for zy_witha, 140, 100
    BATCH_SIZE_TEST=1 #100 for test, 50
    MIN_FREQ=3 #5
    EARLY_STOP_STEP=10
    MAX_LENGTH_EVALUATE=MAX_LENGTH




#changed parameters
SET='continue' #continue, evaluate, moto, alignment, OptAER
CHOOSE_DATASET='test' # dev, test
EVELUAGE_METHOD='greedy_decode' #beam_search greedy_decode
USE_COVERAGE=True
REVERSE_TRANSLATION=True

DEVICE='cuda'
DEVICES=[0,1,2,3]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

#fixed parameters
COMPUTE_MS=False
MAKE_DATA=False
USE_PRETRAINED_EMBEDDING=True
USE_AER_THRESHOLD=False
PARALLEL=True
USE_Tanh=True
USE_LengthNormalization=True

EPOCH=12
EMBED=1000
HIDDEN=1000
MAX_FERTILITY=3
AER_THRESHOLD=0.971106105517593

ALIGN_FILE=MODEL_FILE+'_align.txt'
EVALUATE_FILE=MODEL_FILE+'_evaluate.txt'
TRAIN_FILE=data+SMALL+symbol+'train.tsv'
VALID_FILE=data+SMALL+symbol+'valid.tsv'
TEST_FILE=data+SMALL+symbol+'test_2014.tsv'
TUNE_FILE=TUNE_FILE
TRAINSLATION_FILE = MODEL_FILE+'_translation_'+CHOOSE_DATASET+'.txt'


