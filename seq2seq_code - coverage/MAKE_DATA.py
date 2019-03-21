# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:58:17 2018

@author: zhang
"""
from DefinedParameters import USE_DE2EN_DATA, USE_SMALL, PAIRS_FILE, MAX_LENGTH, TRAIN_FILE, VALID_FILE, TEST_FILE, TUNE_FILE
import pickle
with open(PAIRS_FILE,'rb') as f2: #ATT not all data
    dataset = pickle.load(f2)
print('Dataset size: ', len(dataset))

if USE_DE2EN_DATA:
    '''
    Read 1920209 sentence pairs
    Indexed 417698 words in input language, 149587 words in output'''

    keep_pairs = []
    for pair in dataset:
        if (1 < len(pair[0].split()) <= MAX_LENGTH) and (1 < len(pair[1].split()) <= MAX_LENGTH) :
        #if len(pair[1]) <= MAX_LENGTH :
            keep_pairs.append(pair)
            
    print('only keep', len(keep_pairs),'pairs whose length <=', MAX_LENGTH)
    dataset = keep_pairs
    """## データセットの保存
    データセットを`train`、`validation`、`test`に分割して保存する
    """     
    print('start make data')
    if USE_SMALL:	 
        with open('data/20181226_12_13_14_15_pairs.pk','rb') as f2: #ATT not all data
            pair_2012, pair_2013, pair_2014, pair_2015  = pickle.load(f2)
        print('Dataset size: dev %s, test %s' % (len(pair_2013), len(pair_2015)))
        train = dataset
        valid = pair_2013
        test  = pair_2015
    else:
        # Write out Dataset
        #dataset = dataset[:100]
        import random
        random.seed(0)
        random.shuffle(dataset)
        train = dataset[       0:  640000] #1228000
        valid = dataset[  640000:  800000] #308000
        test  = dataset[  800000: 1000000] #384209
    print(train[0], valid[0], test[0])

else:

     """## データセットの保存
     データセットを`train`、`validation`、`test`に分割して保存する
     """     
     # Write out Dataset
     #dataset = dataset[:100]
                       # clean    all
     train = dataset[       0:  440288] #329882    440288
     valid = dataset[  440288:  441454] #331048    441454
     tune  = dataset[  441454:  442689] #332283    442689
     test  = dataset[  442689:        ] #
     print(train[0], valid[0], tune[0], test[0])
  
     keep_pairs = []
     for pair in train:
         if 0 < len(pair[0].split()) <= MAX_LENGTH and 0 < len(pair[1].split()) <= MAX_LENGTH:
             keep_pairs.append(pair)
     print('only keep', len(keep_pairs),' training data whose length <=', MAX_LENGTH)
     train = keep_pairs
          
                  
def write_tsv(file_name, dataset):
    text = '\n'.join(['\t'.join(pair) for pair in dataset])
    with open(file_name, 'w') as f:
        print(text, file=f)
      
write_tsv(TRAIN_FILE, train)
#write_tsv(VALID_FILE, valid)
if not USE_DE2EN_DATA:
    write_tsv(TUNE_FILE, tune)    
#write_tsv(TEST_FILE, test)
print('data have been saved as files')
