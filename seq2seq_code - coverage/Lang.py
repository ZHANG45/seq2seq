# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:24:39 2018

@author: zhang
"""

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<pad>", 1: "<sos>", 2: "<eos>"}  
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