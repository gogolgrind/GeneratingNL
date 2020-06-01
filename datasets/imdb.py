# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:29:01 2019

@author: HSU, CHIH-CHAO
@Modified by Konstantin Sozykin
"""
import spacy
import torchtext
from torchtext import data, datasets
from torchtext.vocab import GloVe

import re

import pandas as pd
from numpy.random import RandomState

import torchtext
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe
from torchtext.data import Iterator, BucketIterator
import torchtext.datasets

"""
Code taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
"""

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def split_train_valid(path_data='imdb.csv', 
                      path_train='train.csv', path_valid='valid.csv',
                      nrows = 1, 
                      frac=0.8, rng = 0):
    df = pd.read_csv(path_data)
    #rng = 42 #RandomState()
    tr = df.sample(frac=frac, random_state=rng)
    tst = df.loc[~df.index.isin(tr.index)]
    print("Spliting original file to train/valid set...")
    tr.to_csv(path_train, index=False)
    tst.to_csv(path_valid, index=False)

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

spacy_en = spacy.load('en', disable=['tagger', 'parser', 'ner', 'textcat'
                                     'entity_ruler', 'sentencizer', 
                                     'merge_noun_chunks', 'merge_entities',
                                     'merge_subtokens'])

class IMDB_Dataset:

    def __init__(self, emb_dim=50, mbsize=32,
                 path_train = './train.csv',
                 path_valid = './valid.csv',
                 fix_length=32,pretrained = "twitter.27B.50d"):
        self.TEXT = data.Field(sequential=True,init_token='<start>', eos_token='<eos>', 
                               lower=True, tokenize=tokenizer,fix_length=fix_length)
        self.LABEL = data.Field(sequential=False, unk_token=None)
        
        name = ".".join(pretrained.split('.')[:2])
        dim = int(pretrained.split('.')[-1].replace('d',''))

        print('Preprocessing the text...')
        #clean the text
        self.TEXT.preprocessing = torchtext.data.Pipeline(clean_str)

        

        train_datafield = [('text', self.TEXT),  ('label', self.LABEL)]
        self.tabular_train = TabularDataset(path = path_train,  
                                    format= 'csv',
                                    skip_header=True,
                                    fields=train_datafield)
        
        valid_datafield = [('text', self.TEXT),  ('label',self.LABEL)]
        
        self.tabular_valid = TabularDataset(path = path_valid, 
                              format='csv',
                              skip_header=True,
                              fields=valid_datafield)
        
        self.TEXT.build_vocab(self.tabular_train, vectors=GloVe(name,dim))
        self.LABEL.build_vocab(self.tabular_train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim
        

        self.train_iter =Iterator(
            self.tabular_train, 
            batch_size=mbsize,
            device = -1, 
            sort_within_batch=False,
            repeat=False)
        
        self.valid_iter = Iterator(
                self.tabular_valid, 
                batch_size=mbsize,
                device=-1,
                sort_within_batch=False, 
                repeat=False)
        
        self.n_train = len(self.tabular_train)
        self.n_train_batch = self.n_train//mbsize 

        self.n_val = len(self.tabular_valid)

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def get_vocab(self):
        return self.TEXT.vocab

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]