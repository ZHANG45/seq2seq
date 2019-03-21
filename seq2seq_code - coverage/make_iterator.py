# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:25:29 2018

@author: zhang
"""
import torch
from torchtext import data
from DefinedParameters import TRAIN_FILE, VALID_FILE, TEST_FILE, SET, BATCH_SIZE, BATCH_SIZE_TEST, VOCAB_SIZE, DEVICE
# Prepare Data Fields

def MAKE_ITER():
    print('start prepare data file')
    src_TEXT = data.Field(
        sequential=True, include_lengths=True,
        eos_token='<eos>',#init_token='<sos>', 
        #lower=True, tokenize=lambda s: mecab.parse(s).rstrip().split()))
        lower=True) 
    trg_TEXT = data.Field(
        sequential=True, include_lengths=True,
        init_token='<sos>', eos_token='<eos>',
        lower=True) 
        #lower=True, tokenize='spacy') 
    fields = [('src', src_TEXT), ('trg', trg_TEXT)]

    # Load Dataset Using torchtext
    import csv
    csv.field_size_limit(100000000)
    #print(csv.field_size_limit())
    
    train, val, test = data.TabularDataset.splits(
            path='./',
            train=TRAIN_FILE,
            validation=VALID_FILE,
            test=TEST_FILE,
            format='tsv',
            fields=fields,
            csv_reader_params={"quotechar": None})
    
    # Build Vocablary
    src_TEXT.build_vocab(train, max_size=VOCAB_SIZE) #max_size=50000) min_freq=MIN_FREQ)
    trg_TEXT.build_vocab(train, max_size=VOCAB_SIZE)#max_size=50000) 

    
    #if REVERSE_TRANSLATION: #from target to source
    #    src_TEXT, trg_TEXT = trg_TEXT, src_TEXT
    
    # Make iterator
    if SET in ['continue', 'moto']:
        train_iter, valid_iter = data.Iterator.splits(
            (train, val), batch_size=BATCH_SIZE, repeat=False,
            sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)),
            device=torch.device(DEVICE))
        test_iter = None
        # Check Dataset
        print('we have', len(train), 'training dataset')
        print('train',train[0].src, train[0].trg)
        print('val',val[1].src, val[1].trg)
        print('test', test[1].src, test[1].trg)

        
    else:
        train_iter = None
        valid_iter = None
        test_iter = data.Iterator(
                    test, batch_size=BATCH_SIZE_TEST, repeat=False, shuffle=False,device=torch.device(DEVICE))
    
    # Check Vocab
    print('source vocaburary:',len(src_TEXT.vocab.itos))
    print('target vocaburary:',len(trg_TEXT.vocab.itos))
    print(src_TEXT.vocab.itos[:10])
    print(trg_TEXT.vocab.itos[:10])
    return train_iter, valid_iter, test_iter, src_TEXT, trg_TEXT
