#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: att_model.py
@time: 2018/7/14 17:26
"""

import tensorflow as tf
from model.embeding import *
from model.attention import *
from model.cnn_model import *
from model.rnn_model import *
from model.fc_model import *
from utils.utils import *
import numpy as np

class ATT_model():
    def __init__(self,emb_dim, length, vocab_size, filter_size, conv_out, lstm_cell, tfidf_len , layer=1, use_prob=False, opt_method="sgd"):
        self.emb_dim = emb_dim
        self.length = length
        self.vocab_size = vocab_size
        self.filter_size = filter_size
        self.conv_out = conv_out
        self.lstm_cell = lstm_cell
        self.tfidf_len = tfidf_len
        self.cnn_layer = layer
        self.use_prob = use_prob
        self.opt_method = opt_method


    def _creatplacehoder(self):
        self.x1_word = tf.placeholder(dtype=tf.int32, shape=[None, self.length], name = "source_sentence") #source
        self.x1_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.tfidf_len], name = "source_tfidf")
        self.x1_labels = tf.placeholder(dtype=tf.int32, shape=[None, 2], name = "source_labels")

        self.x2_word = tf.placeholder(dtype=tf.int32, shape=[None, self.length], name = "target_sentence") #target
        self.x2_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.tfidf_len], name = "target_tfidf")
        self.x2_labels = tf.placeholder(dtype=tf.int32, shape=[None, 2],name="target_labels")

        self.d_label = tf.placeholder(dtype=tf.int32,shape=[None, 2],name = "domain_labels") #"use crossentropy training domain network"
        self.keep_prob = tf.placeholder(dtype=tf.float32 ,name="drop_out")

        self.dw = tf.placeholder(dtype=tf.float32,name = "FlipGradientWeight")

    def creat_model(self):
        self.embed = embed(vocab_size=self.vocab_size,embed_size=self.emb_dim)
        self.cnn_x1 = CNN_model(name ="cnn_x1")         #input tensor: [batch, length, embed_size] kernel_size[fiter_size,len_size]
        self.cnn_x2 = CNN_model(name ="cnn_x2")
        self.Attention_1 = Attention_1()                               #input tensor: [batch, length-h,embed_size] ,self attention
        self.fc_layer = FC_model(name= "fc_layer")

    def Cnn_layer(self, x, filter, name = "cnn_layer", reuse = False):
        x = self.cnn_x1(x, kernel_size=[filter, self.emb_dim], out_size=300, variable_name=name, reuse=reuse)

        x = self.Attention_1()


    def fc_layer(self,x):
        pass



if __name__ == '__main__':
    pass


