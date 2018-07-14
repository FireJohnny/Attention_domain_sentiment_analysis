#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: model_lstm.py
@time: 2018/6/22 20:57
"""



import tensorflow as tf
from model.embeding import *
from model.attention import *
from model.cnn_model import *
from model.rnn_model import *
from model.fc_model import *
from utils.utils import *
import numpy as np


class cnn_model():
    def __init__(self, emb_dim, length, vocab_size, filter_size, conv_out, lstm_cell, tfidf_len ,layer=1,use_prob =False ):
        self.emb_dim = emb_dim
        self.length = length
        self.vocab_size = vocab_size
        self.filter_size = filter_size
        self.conv_out = conv_out
        self.lstm_cell = lstm_cell
        self.tfidf_len = tfidf_len
        self.cnn_layer = layer
        self.use_prob = use_prob

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
        self.Attention_2 = Attention_1(name = "Attention_2")
        self.att_mat = Attention_2(name = "att_mat")                          #input tensor1: [batch, x_1] input tensor2: [batch, x_2], outside attention before conv
        self.Att_mat_2 = Attention_2(name="att_mat_2")                        #outside attention before full connect
        self.Att_mat_3 = Attention_2(name = "att_mat_3")
        self.fc_layer = FC_model(name= "fc_layer")

    def CNN_layer(self,variable_scope, x1, x2):

        with tf.variable_scope(variable_scope):
            #shared cnn weighted
            L_conv = self.cnn_x1(self.pad_(x1), kernel_size=(self.emb_dim, self.filter_size), pad="VALID", reuse = False)
            # R_conv = self.cnn_x2(x2, kernel_size=(self.filter_size, self.emb_dim),reuse = False)
            R_conv = self.cnn_x1(self.pad_(x2), kernel_size=(self.emb_dim, self.filter_size), pad="VALID", reuse = True)
            L_conv = tf.transpose(L_conv, [0, 3, 2, 1])
            R_conv = tf.transpose(R_conv, [0, 3, 2, 1])

            att_mat = self.Att_mat_2(L_conv, R_conv)
            L_attention, R_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)
            #pool
            L_wp = w_pool(variable_scope="left", x=L_conv, attention=L_attention, model_type="CNN", length=self.length, filer_size=self.filter_size)
            L_ap = all_pool("left", L_conv, length=self.length, filter_size=self.filter_size,embed_szie=self.emb_dim,conv_out=64)
            R_wp = w_pool(variable_scope="right", x=R_conv, attention=R_attention, model_type="CNN", length=self.length, filer_size=self.filter_size)
            R_ap = all_pool("left", R_conv, length=self.length, filter_size=self.filter_size,embed_szie=self.emb_dim,conv_out=64)

            return L_wp, L_ap, R_wp, R_ap

    def squeeze_data(self,x):
        x = [tf.squeeze(i,[1]) for i in tf.split(x,self.length,1)]
        return x

    def LSTM_layer(self, variable_scope, x1, x2):
        with tf.variable_scope(variable_scope):
            x1 = self.squeeze_data(x1)
            x2 = self.squeeze_data(x2)

    def input_att(self, x1, x2):
        with tf.variable_scope("att_amt"):
            aW = tf.get_variable(name = "aW", shape=(self.length, self.emb_dim))
            att_mat = self.att_mat(x1, x2)
            x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, aW)), -1)
            x2_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", tf.transpose(att_mat, [0, 2, 1]), aW)), -1)

            x1 = tf.concat([x1, x1_a], axis = 3)
            x2 = tf.concat([x2, x2_a], axis = 3)
        return x1,x2

    def pad_(self, x):
        x = tf.pad(tensor=x, paddings=np.array([[0, 0],[0, 0], [self.filter_size-1, self.filter_size -1], [0, 0]]),name = "wide_pad" )
        return x
    def tfidf_fc(self,x, reuse =False):
        x = self.fc_layer(x,unit = 1000, activation = tf.nn.relu,variable_name="fc_layer_1",reuse = reuse)
        x = self.fc_layer(x,unit = 500, activation = tf.nn.relu, variable_name="fc_layer_2",reuse = reuse)
        x = self.fc_layer(x,unit = 200, activation = tf.nn.relu, variable_name="fc_layer_3",reuse = reuse)
        return x
    def sentiment_fc(self,x, reuse =False):
        x = self.fc_layer(x,unit = 500, activation = tf.nn.relu, variable_name="senti_layer_1", reuse =reuse)
        x = self.fc_layer(x,unit = 200, activation = tf.nn.relu, variable_name="senti_layer_2", reuse = reuse)
        x = self.fc_layer(x,unit = 2,activation = None, variable_name="senti_layer_3", reuse = reuse)
        return x

    def domain_fc(self, x, reuse=False):

        x = flip_gradient(x,self.dw)
        x = self.fc_layer(x,unit = 200,activation = tf.nn.relu,variable_name="domain_layer_1", reuse = reuse)
        x = self.fc_layer(x,unit = 2,activation = None,variable_name="domain_layer_2", reuse = reuse)
        return x

    def compute_loss(self,senti,domain_x1,domain_x2):
        senti_loss = tf.losses.softmax_cross_entropy(self.x1_labels,senti)
        domain_loss = -tf.reduce_mean(domain_x1) + tf.reduce_mean(domain_x2)

        return senti_loss+domain_loss


    def build(self):
        self._creatplacehoder()
        self.creat_model()
        with tf.name_scope("word_embed"):
            x1 = self.embed(self.x1_word)
            x2 = self.embed(self.x2_word,reuse=True)
        with tf.device("/gpu:0"),tf.name_scope("input_att"):
            if self.use_prob:
                x1 = tf.nn.dropout(x1,self.keep_prob)
                x2 = tf.nn.dropout(x2,self.keep_prob)
            x1, x2 = self.input_att(x1, x2)

        L_wp,R_wp = x1,x2
        for i in range(self.cnn_layer):
            with tf.device("/gpu:2"), tf.name_scope("convolution_".format(i)):
                if self.use_prob:
                    L_wp = tf.nn.dropout(L_wp,self.keep_prob)
                    R_wp = tf.nn.dropout(R_wp,self.keep_prob)
                L_wp, L_ap, R_wp, R_ap = self.CNN_layer("Cnn_layer", L_wp, R_wp)

        with tf.device("/gpu:1"),tf.name_scope("tfidf_process"):
            if self.use_prob:
                self.x1_tf = tf.nn.dropout(self.x1_tf,self.keep_prob)
                self.x2_tf = tf.nn.dropout(self.x2_tf,self.keep_prob)
            x1_tf = self.tfidf_fc(self.x1_tf, reuse=False)
            x2_tf = self.tfidf_fc(self.x2_tf, reuse=True)
        #concat tf featuer and cnn feature
        x1_f = tf.concat([L_ap, x1_tf], axis=1)
        x2_f = tf.concat([R_ap, x2_tf], axis=1)
        if self.use_prob:
            x1_f = tf.nn.dropout(x1_f,self.keep_prob)
            x2_f = tf.nn.dropout(x2_f,self.keep_prob)

        with tf.name_scope("sentiment_analysis"):
            senti_x1 = self.sentiment_fc(x1_f)
            senti_x2 =self.sentiment_fc(x2_f,reuse=True)
        with tf.name_scope("domain_adaption"):
            domain_x1 = self.domain_fc(x1_f)
            domain_x2 = self.domain_fc(x2_f, reuse = True)

        self.loss = self.compute_loss(senti_x1, domain_x1, domain_x2)
        self.optimizer()
        x1_p = self.pred(senti_x1,)
        x2_p = self.pred(senti_x2)
        x1_r = self.pred(self.x1_labels)
        x2_r = self.pred(self.x2_labels)
        self.acc_x1 = self.acc(x1_r, x1_p)
        self.acc_x2 = self.acc(x2_r, x2_p)
        self.Summary()
    def optimizer(self):
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss)

    def acc(self,r,p):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(r,p),dtype=tf.float32))
        return accuracy

    def pred(self,x):
        p = tf.argmax(x,axis=1)
        return p

    def Summary(self):
        tf.summary.scalar("source_acc", self.acc_x1)
        tf.summary.scalar("source_acc", self.acc_x2)
        tf.summary.scalar("loss", self.loss)
        self.summary = tf.summary.merge_all()

if __name__ == '__main__':
    m = cnn_model(300,100,5000,2,32,32,2000,1)
    m.build()
    pass


