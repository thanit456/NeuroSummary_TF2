from services.inferencer import Inferencer
from config.hparams import create_hparams
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessed.preprocessed_thaigov import word_to_index, preprocess_text
import tensorflow as tf
import pickle
import numpy as np

import pythainlp
from tqdm import tqdm_notebook
import pandas as pd

import unicodedata
import re
import os
import io
import time

with open('./decode/dictionary/extractive/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

with open('./decode/dictionary/extractive/idx2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)

def convert_to_sequence(a, word2idx):
    ans = []
    for item in a:
        try:
            ans.append(word2idx[item])
        except:
            ans.append(word2idx['<UNK>'])
    return ans
    
def convert_to_text(a):
    s = []
    for item in a:
        try:
            s.append(idx2word[item])
        except:
            s.append('<UNK>')
    return s

def preprocess_data(content):
    _X = []
    for j in range(0, len(content), 5):
        _t = convert_to_sequence(content[j: j+5], word2idx)
        _X.append(_t)
    return _X

class WordEncoder(tf.keras.Model):
    
    def __init__(self, batch_size=128, hidden_units=200):
        super(WordEncoder, self).__init__()

        with open('./model/extractive/fasttext_vec_2.pkl', 'rb') as f:
            W = pickle.load(f)

        self.hidden_units = hidden_units
        self.batch_size = batch_size

        self.emb = tf.keras.layers.Embedding(len(word2idx), 300, weights=[W])
        self.bigru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.hidden_units, return_sequences=True, return_state=False))
        self.dense1 = tf.keras.layers.Dense(100, activation='tanh')

    def call(self, x):
        outputs = []
        for i in range(len(x)):
            _x = self.emb(x[i])
            output = self.bigru1(_x, initial_state=[tf.zeros((_x.shape[0], self.hidden_units)),tf.zeros((_x.shape[0], self.hidden_units))])
            # print('A',output.shape)
            output = tf.reduce_mean(output, axis=1, keepdims=False) / 5
            output = self.dense1(output)
            outputs.append(output)
        return outputs

class SentenceEncoder(tf.keras.Model):
    
    def __init__(self, batch_size=64, hidden_units=200):
        super(SentenceEncoder, self).__init__()
        self.hidden_units = hidden_units
        self.batch_size = batch_size

        self.bigru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.hidden_units, return_sequences=True, return_state=False))
        self.dense1 = tf.keras.layers.Dense(100, activation='tanh')

    def call(self, x):
        outputs = []
        for i in range(len(x)):
            # print(x[i].shape)
            #initial state shape = (batch_size, hidden_units)
            output = self.bigru1(tf.expand_dims(x[i],0), initial_state=[tf.zeros((1, self.hidden_units)),tf.zeros((1, self.hidden_units))])
            # print('B',output.shape)
            output = tf.reduce_mean(output, axis=1, keepdims=False) / x[i].shape[0]
            output = self.dense1(output)
            outputs.append(output)
        return outputs

class SentenceProbability(tf.keras.Model):
    def __init__(self, batch_size=64):
        super(SentenceProbability, self).__init__()
        self.batch_size = batch_size

        w_init = tf.random_normal_initializer()
        self.w_c = tf.Variable(initial_value=w_init(shape=(100, 1),dtype='float32'),trainable=True)
        self.w_s = tf.Variable(initial_value=w_init(shape=(100, 100),dtype='float32'),trainable=True)
        self.w_r = tf.Variable(initial_value=w_init(shape=(100, 100),dtype='float32'),trainable=True)

        self.pos_emb = tf.keras.layers.Embedding(3000, 100)
        self.w_ap = tf.Variable(initial_value=w_init(shape=(100, 1),dtype='float32'),trainable=True)

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(1,),dtype='float32'),trainable=True)


    def call(self, h, d):
        P = []
        for i in range(len(d)):
            content = tf.matmul(h[i], self.w_c)
            # print('content', content.shape)
            _t = tf.matmul(d[i],self.w_s)
            # print('_t', _t.shape)
            salience = tf.matmul(h[i], tf.transpose(_t))
            # print('salience',salience.shape)
            emb = self.pos_emb(tf.range(h[i].shape[0]))
            # print('emb', emb.shape)
            abs_pos_imp = tf.matmul(emb, self.w_ap)
            # print('abs_pos_imp', abs_pos_imp.shape)

            novelty = 0
            p = [] 
            for j in range(h[i].shape[0]):
                if j == 0:
                    s = tf.zeros((1, 100))
                    p.append(tf.math.sigmoid(content[j]+salience[j]+abs_pos_imp[j]+self.b)[0])
                else:
                    # print('s', s.shape)
                    _k = tf.matmul(tf.math.tanh(s),self.w_r)
                    # print('_k', _k.shape)
                    _h = tf.expand_dims(h[i][j],0)
                    # print('_h', _h.shape)
                    novelty = tf.matmul(_h, tf.transpose(_k))
                    # print('novelty', novelty.shape)
                    p.append(tf.math.sigmoid(content[j]+salience[j]-novelty+abs_pos_imp[j]+self.b)[0,0])
                    # print('p(y=1):', p[-1])
                    s += _h *  p[-1]
            # print(tf.stack(p, axis=0))
            P.append(tf.stack(p, axis=0))
        return P

def init_encoder():
    word_encoder = WordEncoder(64, 200)
    sent_encoder = SentenceEncoder(64, 200)
    sentence_prob = SentenceProbability(64)

    sample_outputs = word_encoder([np.zeros((76, 5))])
    sample_outputs2 = sent_encoder(sample_outputs)
    sample_outputs3 = sentence_prob(sample_outputs, sample_outputs2)

    with open('./model/extractive/word_encoder_weights.pkl', 'rb') as f:
        word_encoder.set_weights(pickle.load(f))

    with open('./model/extractive/sent_encoder_weights.pkl', 'rb') as f:
        sent_encoder.set_weights(pickle.load(f))
        
    with open('./model/extractive/sentence_prob_weights.pkl', 'rb') as f:
        sentence_prob.set_weights(pickle.load(f))
    
    return word_encoder, sent_encoder, sentence_prob

def model_predict(st, word_encoder, sent_encoder, sentence_prob):
    sen = pythainlp.tokenize.word_tokenize(st, engine='deepcut')
    sen2 = list(sen)
    for i in range(len(sen)):
        try:
            sen[i] = word2idx[sen[i]]
        except:
            sen[i] = word2idx['<UNK>']
    sentences = []
    ans = []
    for i in range(0, len(sen), 5):
        sentences.append(sen[i: i+5])
        ans.append(sen2[i: i+5])
    inp = np.array(pad_sequences(sentences, padding='post'))
    # print(inp, inp.shape)
    h = word_encoder([inp])
    d = sent_encoder(h)
    #calculate P(y_j = 1 | h_j, s_j, d) for each docs
    p = np.array(sentence_prob(h, d))[0]
    choose = sorted([(i, p[i]) for i in range(len(p))],key=lambda tup: tup[1])
    sen = []
    for c in sorted(choose[:5],key=lambda tup: tup[0]):
        sen += ans[c[0]]
    return ''.join(sen)

class ExtractiveInferencer(Inferencer):

    def __init__(self):
        super().__init__()
        
        word_encoder, sent_encoder, sentence_prob = init_encoder()
        self.word_encoder = word_encoder
        self.sent_encoder = sent_encoder
        self.sentence_prob = sentence_prob

    def get_name(self):
        return 'Extractive'

    def preprocess(self, content):
        tokenized = preprocess_text(content)
        return preprocess_data(tokenized)

    def infer(self, content):
        return model_predict(content, self.word_encoder, self.sent_encoder, self.sentence_prob)