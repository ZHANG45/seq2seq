# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:58:17 2018

@author: zhang
"""
from DefinedParameters import TRAIN_FILE, VALID_FILE, TEST_FILE, train_src_file, train_trg_file, valid_src_file, valid_trg_file, test_src_file, test_trg_file

def read_data(src_file, trg_file):
    src_data, trg_data = [] , []
    with open(src_file, encoding='utf-8') as src,  open(trg_file, encoding='utf-8') as trg:
        src_data += src.readlines() 
        trg_data += trg.readlines()
    pairs = [[src_line, trg_line] for src_line, trg_line in zip(src_data, trg_data)]
    return pairs

def write_tsv(file_name, dataset):
    text = '\n'.join(['\t'.join(pair) for pair in dataset])
    with open(file_name, 'w') as f:
        print(text, file=f)
        
print('start make data')
        
train =  read_data(train_src_file, train_trg_file)
valid =  read_data(valid_src_file, valid_trg_file)
test  =  read_data(test_src_file, test_trg_file)
print(train[0], valid[0], test[0])  

write_tsv(TRAIN_FILE, train)
write_tsv(VALID_FILE, valid)
write_tsv(TEST_FILE, test)
print('data have been saved as files')
