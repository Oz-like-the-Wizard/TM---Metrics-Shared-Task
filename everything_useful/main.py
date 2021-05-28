from keras.layers import Dense, Dropout
from keras import Sequential
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer

import numpy as np
import os
import string 
import re 
from nltk.tokenize import MWETokenizer, word_tokenize
from tqdm import tqdm
import spacy
from gensim.models import KeyedVectors

def ngram_calc(pt, gt, mode="precision", ngram):
    ptgram = [" ".join(pt[i:i+ngram]) for i in range(len(pt[:len(pt)-ngram+1]))]
    gtgram = [" ".join(gt[i:i+ngram]) for i in range(len(gt[:len(gt)-ngram+1]))]
    if mode == "precision":
        den = len(ptgram)
        nom = 0
        for j in ptgram:
            if j in gtgram:
                nom += 1
                gtgram.remove(j)
        try:
            tot = nom/den
        except ZeroDivisionError:
             tot = 0
        return tot
    elif mode == "recall":
        den = len(gtgram)
        nom = 0
        for j in gtgram:
            if j in ptgram:
                nom += 1
                ptgram.remove(j)
        try:
            tot = nom/den
        except ZeroDivisionError:
             tot = 0
        return tot

def create_model(neuron_list, dropout=False, dropout_rate=0.2, opt="adam", **kwargs):
    model = Sequential()
    model.add(Dense(neuron_list[1], activation="relu", input_dim=neuron_list[0]))
    for i in neuron_list[2:-1]:
        model.add(Dense(i, activation="relu", **kwargs))
    if dropout ==True:
        model.add(Dropout(dropout_rate))
    model.add(Dense(neuron_list[-1], ))
    #Compile
    model.compile(optimizer=opt, loss="mean_squared_error", metrics=["accuracy"])
    return model


def not_that_clean(text_list, lemmatize, stemmer):
    """
    Function that a receives a list of strings and preprocesses it.
    
    :param text_list: List of strings.
    :param lemmatize: Tag to apply lemmatization if True.
    :param stemmer: Tag to apply the stemmer if True.
    """
    stop = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()
    snowball_stemmer = SnowballStemmer('english')
    
    updates = []
    for j in range(len(text_list)):
        
        text = text_list[j]

        #REMOVE THAT IS NOT TEXT
        text = re.sub("[^a-zA-Z]", '', text)

        text = word_tokenize(text)

        # REMOVE STOP WORDS
        #text = [word for word in text if word not in stop]

        #LEMMATIZATION
        if lemmatize:
            text = [lemma.lemmatize(word) for word in text]
        
        #STEMMER
        if stemmer:
            text = [snowball_stemmer.stem(word) for word in text.split()]
        
        updates.append(text)
        
    return updates

def tokenization(corpus, tool, language = 'en', pos_to_remove = ['PUNCT','NUM'], ent_to_remove = ['PERSON','ORG'], stop_words_to_remove= False, lowercase = True, regex=True):
    """
    tool: one of two strings - 'spacy' or 'NLTK'
    languages (string ISO code): supports 'en', 'fi' or 'zh'
    pos_to_remove (list): part-of-speech tag from spacy
    ent_to_remove (list): entities from spacy
    
    
    """
    tokenized_corpus = []
    if tool == 'spacy':
        if language == 'en':
            sc = spacy.load('en_core_web_sm')
            for doc in tqdm(sc.pipe(corpus, disable=["lemmatizer", "textcat", "custom"])):
                if stop_words_to_remove:
                    doc_list = [word.text for word in doc if word.pos_ not in pos_to_remove if word.ent_type_ not in ent_to_remove if word.is_stop]
                else: 
                    doc_list = [word.text for word in doc if word.pos_ not in pos_to_remove if word.ent_type_ not in ent_to_remove]

                if lowercase:
                    doc_list = [word.lower() for word in doc_list]

                tokenized_corpus.append(doc_list)
        if language == 'zh':
            sc = spacy.load('zh_core_web_sm')
            for doc in tqdm(sc.pipe(corpus, disable=["lemmatizer", "textcat", "custom"])):
                if stop_words_to_remove:
                    doc_list = [word.text for word in doc if word.pos_ not in ['PUNCT'] if word.is_stop]
                else: 
                    doc_list = [word.text for word in doc if word.pos_ not in ['PUNCT']]
                
                tokenized_corpus.append(doc_list)
        if language == 'fi':
            sc = spacy.load('xx_sent_ud_sm')
            for doc in tqdm(sc.pipe(corpus, disable=["lemmatizer", "textcat", "custom"])): 
                doc_list = [word.text for word in doc]
                
                if lowercase:
                    doc_list = [word.lower() for word in doc_list]

                tokenized_corpus.append(doc_list)
                
            
        
    if tool == 'NLTK':
        print('Not implemented')
        
        
    return tokenized_corpus


def match_regex(tokenized_corpus, letters = True, letters_and_numbers = False):
    
    if letters:
        regex = r'[a-z]+'
    if letters_and_numbers:
        regex = r'([a-z]+|^\d+$)'
        
    new_tokenized = []
    for sentence_list in tokenized_corpus:
        sentence_list2 = [word for word in sentence_list if re.search(regex, word)]
        new_tokenized.append(sentence_list2)
        
    return new_tokenized

def chinese_regex(tokenized_corpus, chinese = True, chinese_and_numbers = False):
    
    if chinese:
        regex = r'[\u4e00-\u9fff]+'
    if chinese_and_numbers:
        regex = r'([\u4e00-\u9fff]+|^\d+$)'
        
    new_tokenized_zh = []
    for sentence_list in tokenized_corpus:
        chinese = [word for word in sentence_list if re.search(regex, word)]
        new_tokenized_zh.append(chinese)
        
    return new_tokenized_zh

def the_preprocessor(df,lang):
    if lang == "en":
        word_vect = KeyedVectors.load('en_vectors.kv')

        df["pt"] = tokenization(df["translation"], tool = 'spacy', language = lang, pos_to_remove = ['PUNCT', 'NUM'], ent_to_remove = [], stop_words_to_remove= False, lowercase = True)

        df["ref"] = tokenization(df["reference"], tool = 'spacy', language = lang, pos_to_remove = ['PUNCT', 'NUM'], ent_to_remove = [], stop_words_to_remove= False, lowercase = True)

    elif lang == "fi":
        word_vect = KeyedVectors.load('fi_vectors.kv')

        df["pt"] = tokenization(df["translation"], tool = 'spacy', language = lang,lowercase = True)
        df["pt"] = match_regex(df["pt"], letters = True, letters_and_numbers = False)

        df["ref"] = tokenization(df["reference"], tool = 'spacy', language = lang,lowercase = True)
        df["ref"] = match_regex(df["ref"], letters = True, letters_and_numbers = False)

    elif lang == "zh":
        word_vect = KeyedVectors.load('zh_vectors.kv')

        df["pt"] = tokenization(df["translation"], tool = 'spacy', language = 'zh', stop_words_to_remove= False)

        df["ref"] = tokenization(df["reference"], tool = 'spacy', language = 'zh', stop_words_to_remove= False)

    else:
        return "ooppsiiee, language not defined"

    for mode in ["precision", "recall"]:
        for i in range(1,5):
            df[str(i)+"gram-"+ mode] = df.apply(lambda x: ngram_calc(x.pt, x.ref, mode=mode, ngram=i), axis =1)

    df['wmdist'] = df.apply(lambda x: word_vect.wmdistance(x["pt"],x["ref"]), axis=1)
    
    df["pt_len"] = df['pt'].str.len()
    df["ref_len"] = df['ref'].str.len()

    df = df.drop(["reference", "translation", "avg-score", "annotators"], axis=1)
    
    return df


