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
import re 
from nltk.tokenize import MWETokenizer, word_tokenize
from tqdm import tqdm
import spacy
from gensim.models import KeyedVectors

def ngram_calc(pt, gt, mode="precision", ngram=1):
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

def create_model(neuron_list, dropout=False, dropout_rate=0.2, opt="adam", loss="mse", **kwargs):
    model = Sequential()
    model.add(Dense(neuron_list[1], activation="relu", kernel_initializer='normal', input_dim=neuron_list[0]))
    for i in neuron_list[2:-1]:
        model.add(Dense(i, kernel_initializer='normal', activation="relu", **kwargs))
    if dropout ==True:
        model.add(Dropout(dropout_rate))
    model.add(Dense(neuron_list[-1]))
    #Compile
    model.compile(optimizer=opt, loss=loss, metrics=["mae"])
    return model

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
                    doc_list = [word.text for word in doc if word.pos_ not in pos_to_remove if word.ent_type_ not in ent_to_remove if not word.is_stop]
                else: 
                    doc_list = [word.text for word in doc if word.pos_ not in pos_to_remove if word.ent_type_ not in ent_to_remove]

                if lowercase:
                    doc_list = [word.lower() for word in doc_list]

                tokenized_corpus.append(doc_list)
        if language == 'zh':
            sc = spacy.load('zh_core_web_sm')
            for doc in tqdm(sc.pipe(corpus, disable=["lemmatizer", "textcat", "custom"])):
                if stop_words_to_remove:
                    doc_list = [word.text for word in doc if word.pos_ not in ['PUNCT'] if not word.is_stop]
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

def the_preprocessor(df,lang, tool="spacy", pos_list=[], ent_list=[], stopwords=False, lowercase=True, let=True, letandnum=False):
    if lang == "en":
        word_vect = KeyedVectors.load('en_vectors.kv')

        df["pt"] = tokenization(df["translation"], tool = tool, language = lang, pos_to_remove = pos_list, ent_to_remove = ent_list, stop_words_to_remove= stopwords, lowercase = lowercase)

        df["ref"] = tokenization(df["reference"], tool = tool, language = lang, pos_to_remove = pos_list, ent_to_remove = ent_list, stop_words_to_remove= stopwords, lowercase = lowercase)

    elif lang == "fi":
        word_vect = KeyedVectors.load('fi_vectors.kv')

        df["pt"] = tokenization(df["translation"], tool = tool, language = lang, lowercase = lowercase)
        df["pt"] = match_regex(df["pt"].tolist(), letters = let, letters_and_numbers = letandnum)

        df["ref"] = tokenization(df["reference"], tool = tool, language = lang, lowercase = lowercase)
        df["ref"] = match_regex(df["ref"].tolist(), letters = let, letters_and_numbers = letandnum)

    elif lang == "zh":
        word_vect = KeyedVectors.load('zh_vectors.kv')

        df["pt"] = tokenization(df["translation"].tolist(), tool = tool, language = lang, stop_words_to_remove= stopwords)
        df["pt"] = chinese_regex(df["pt"].tolist(), chinese= let, chinese_and_numbers = letandnum)

        df["ref"] = tokenization(df["reference"].tolist(), tool = tool, language = lang, stop_words_to_remove= stopwords)
        df["ref"] = chinese_regex(df["ref"].tolist(), chinese= let, chinese_and_numbers = letandnum)

    else:
        return "ooppsiiee, language not defined"

    for mode in ["precision", "recall"]:
        for i in range(1,5):
            df[str(i)+"gram-"+ mode] = df.apply(lambda x: ngram_calc(x.pt, x.ref, mode=mode, ngram=i), axis =1)

    df['wmdist'] = df.apply(lambda x: word_vect.wmdistance(x["pt"],x["ref"]), axis=1)
    
    df["pt_len"] = df['pt'].str.len()
    df["ref_len"] = df['ref'].str.len()
    df["len_diff"] = df["pt_len"] - df["ref_len"]

    df = df.drop(["reference", "translation", "annotators"], axis=1)
    
    return df

def tokenization_lemma(corpus, tool, language = 'en', pos_to_remove = ['PUNCT','NUM'], ent_to_remove = ['PERSON','ORG'], stop_words_to_remove= False, lowercase = True, regex=True):
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
            for doc in tqdm(sc.pipe(corpus, disable=["textcat", "custom"])):
                if stop_words_to_remove:
                    doc_list = [word.lemma_ for word in doc if word.pos_ not in pos_to_remove if word.ent_type_ not in ent_to_remove if not word.is_stop]
                else: 
                    doc_list = [word.lemma_ for word in doc if word.pos_ not in pos_to_remove if word.ent_type_ not in ent_to_remove]

                if lowercase:
                    doc_list = [word.lower() for word in doc_list]

                tokenized_corpus.append(doc_list)
        if language == 'zh':
            sc = spacy.load('zh_core_web_sm')
            for doc in tqdm(sc.pipe(corpus, disable=["textcat", "custom"])):
                if stop_words_to_remove:
                    doc_list = [word.lemma_ for word in doc if word.pos_ not in ['PUNCT'] if not word.is_stop]
                else: 
                    doc_list = [word.lemma_ for word in doc if word.pos_ not in ['PUNCT']]
                
                tokenized_corpus.append(doc_list)
        if language == 'fi':
            sc = spacy.load('xx_sent_ud_sm')
            for doc in tqdm(sc.pipe(corpus, disable=["textcat", "custom"])): 
                doc_list = [word.lemma_ for word in doc]
                
                if lowercase:
                    doc_list = [word.lower() for word in doc_list]

                tokenized_corpus.append(doc_list)
    
        
        
    return tokenized_corpus


def the_preprocessor_lemma(df,lang, tool="spacy", pos_list=[], ent_list=[], stopwords=False, lowercase=True, let=True, letandnum=False):
    if lang == "en":
        word_vect = KeyedVectors.load('en_vectors_lemma.kv')

        df["pt"] = tokenization_lemma(df["translation"], tool = tool, language = lang, pos_to_remove = pos_list, ent_to_remove = ent_list, stop_words_to_remove= stopwords, lowercase = lowercase)

        df["ref"] = tokenization_lemma(df["reference"], tool = tool, language = lang, pos_to_remove = pos_list, ent_to_remove = ent_list, stop_words_to_remove= stopwords, lowercase = lowercase)

    elif lang == "fi":
        word_vect = KeyedVectors.load('fi_vectors_lemma.kv')

        df["pt"] = tokenization_lemma(df["translation"], tool = tool, language = lang, lowercase = lowercase)
        df["pt"] = match_regex(df["pt"].tolist(), letters = let, letters_and_numbers = letandnum)

        df["ref"] = tokenization_lemma(df["reference"], tool = tool, language = lang, lowercase = lowercase)
        df["ref"] = match_regex(df["ref"].tolist(), letters = let, letters_and_numbers = letandnum)

    elif lang == "zh":
        word_vect = KeyedVectors.load('zh_vectors_lemma.kv')

        df["pt"] = tokenization_lemma(df["translation"].tolist(), tool = tool, language = lang, stop_words_to_remove= stopwords)
        df["pt"] = chinese_regex(df["pt"].tolist(), chinese= let, chinese_and_numbers = letandnum)

        df["ref"] = tokenization_lemma(df["reference"].tolist(), tool = tool, language = lang, stop_words_to_remove= stopwords)
        df["ref"] = chinese_regex(df["ref"].tolist(), chinese= let, chinese_and_numbers = letandnum)

    else:
        return "ooppsiiee, language not defined"

    for mode in ["precision", "recall"]:
        for i in range(1,5):
            df[str(i)+"gram-"+ mode] = df.apply(lambda x: ngram_calc(x.pt, x.ref, mode=mode, ngram=i), axis =1)

    df['wmdist'] = df.apply(lambda x: word_vect.wmdistance(x["pt"],x["ref"]), axis=1)
    
    df["pt_len"] = df['pt'].str.len()
    df["ref_len"] = df['ref'].str.len()
    df["len_diff"] = df["pt_len"] - df["ref_len"]

    df = df.drop(["reference", "translation", "annotators"], axis=1)
    
    return df
