# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:11:22 2018

@author: zhang
"""

import pickle
from gensim.models import Word2Vec
import os
from collections import OrderedDict
import nltk
PAD_token = 0
SOS_token = 1
EOS_token = 2
embedding_size = 1000
stop_words = ['\n']
USE_JAPAN = False
READ_PARIS = True
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = OrderedDict({0: "<pad>", 1: "<sos>", 2: "<eos>"})
        self.n_words = 3 # Count default tokens
        self.weight = []

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def split_de(de_data,en_data):    
    new_de = []
    new_en = []
    
    for de, en in zip(de_data, en_data):
        if ('http' in de) or ('http' in en):
            continue
        elif de != en:
            new_de.append(de)
            new_en.append(en)
    print(len(de_data), len(new_de))
    file_address = "temporary.txt"
    n = 100000
    times = int(len(new_de) / n)
    result = []
    
    for j in range(1, times+2):
        start = (j - 1) * n 
        if j * n > len(new_de):
            end = len(new_de)
        else:
            end = j * n
    
        with open(file_address, 'w', encoding='utf-8') as f:
            for i in range(start, end): 
                sentence = new_de[i]
                f.write(sentence+' zy_end\n')  
                    
        cmd = "java -Dfile.encoding=UTF-8 -jar jwordsplitter-4.4.jar "+file_address
        rs = os.popen(cmd)
        cmdout = rs.read()
    
        sentence = []
        for token in cmdout.split('\n'):
            if ', ' in token:
                tokens = token.split(', ')
                sentence += tokens
            elif 'zy_end' in token:
                result.append(' '.join(sentence))
                sentence = []
            else:
                sentence.append(token)
    if len(new_en) != len(result):
        print('%s , %s, length error'%(len(result), len(new_en)))
        exit()
    
    print(len(result),'german data has been splited')
    return result, new_en

def read_langs(lang1, lang2):
    print("Reading lines...")

    # Read the file and split into lines
    trg_data = []
    src_data = []
    if READ_PARIS:
        from DefinedParameters import PAIRS_FILE
        with open(PAIRS_FILE,'rb') as f2: #ATT not all data
            dataset = pickle.load(f2)
        print('Dataset size: ', len(dataset))
    else:
        if USE_JAPAN:
            file_address = 'word_alignment_file/word_alignment.0/kftt-data-1.0/data/toku3000/kyoto-'
            for dataset in ['train','dev','tune','test']:
                f_en = open(file_address+dataset+'.en') #("/home/lr/zhang/europarl-v7/europarl-v7.de-en.en","r",encoding='utf-8')
                f_ja = open(file_address+dataset+'.ja')
                fen_data = f_en.readlines()
                fja_data = f_ja.readlines()  
                trg_data += fen_data
                src_data += fja_data
                #print(dataset, 'end in', len(en_data))
    

                
        else:
            file_address = '/home/lr/zhang/europarl-v7/'
            for dataset in ['train']: #'commoncrawl.de-en', 'europarl-v7.de-en', 'news-commentary-v9.de-en'
                f_de = open(file_address+dataset+'.de', encoding='utf-8')
                f_en = open(file_address+dataset+'.en',encoding='utf-8')

                #fde_data, fen_data = split_de(f_de.readlines(), f_en.readlines())
                src_data += f_de.readlines() 
                trg_data += f_en.readlines()
                f_de.close()
                f_en.close()
           

    train_pairs = []
    input_list = []
    target_list = []
    
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    if not READ_PARIS:
        for src, trg in zip(src_data, trg_data):
            if USE_JAPAN:
                src_part = [word for word in src.lower().split() if word not in stop_words]
                trg_part = [word for word in trg.lower().split() if word not in stop_words]
            elif ('you' not in src.lower()) and ('I' not in src.lower()) and ('we' not in src.lower()) :
                trg_part = [word for word in trg.lower().split() if word not in stop_words]
                src_part = [word for word in src.lower().split() if word not in stop_words]

            train_pairs.append([src, trg])
            input_list.append(['<sos>'] + src_part + ['<pad>','<eos>'])
            target_list.append(['<sos>'] + trg_part + ['<pad>','<eos>'])   
            input_lang.index_words(src_part)
            output_lang.index_words(trg_part)    
        

    else:
        train_pairs = dataset
        for pair in dataset:
            trg_part = pair[1].split()
            src_part = pair[0].split()
            input_list.append(['<sos>'] + src_part + ['<pad>','<eos>'])
            target_list.append(['<sos>'] + trg_part + ['<pad>','<eos>'])   
            input_lang.index_words(src_part)
            output_lang.index_words(trg_part)     

    print("Read %d sentence pairs" % len(train_pairs))   
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print(train_pairs[10])
    print('start training word embedding')     
    input_lang_model = Word2Vec(input_list,size=embedding_size,min_count=1)
    output_lang_model = Word2Vec(target_list,size=embedding_size,min_count=1) 
    print('finished word embedding')  

    print('start insert weight')
    for (k,v) in input_lang.index2word.items(): 
        input_lang.weight.append(input_lang_model.wv[v])
    for (k,v) in output_lang.index2word.items(): 
        output_lang.weight.append(output_lang_model.wv[v])
    print('weight has been inserted')

    return input_lang, output_lang, train_pairs


input_lang, output_lang, pairs = read_langs('ja', 'en')
print('example:', pairs[1])
file_address = 'data/'
pickle_file1 = file_address+'20190112_standford1000d_lang.pk'
f1 = open(pickle_file1, 'wb')
pickle.dump((input_lang, output_lang), f1)

if not READ_PARIS:
    pickle_file2 = file_address+'20181226standford_pairs.pk'
    f2 = open(pickle_file2, 'wb')
    pickle.dump(pairs, f2)
    print('\n embedding data has been dumped into ', pickle_file1)
else:
    print('\n data has been dumped into ', pickle_file1, pickle_file2)