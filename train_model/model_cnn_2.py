#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: model_cnn_2.py
@time: 2018/7/12 10:43
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
    def __init__(self, emb_dim, length, vocab_size, filter_size, conv_out, lstm_cell, tfidf_len , layer=1, use_prob=False, opt_method="sgd"):
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

        self.d_label = tf.placeholder(dtype=tf.int32,shape=[None, 2],name = "domain_labels")  #"use crossentropy training domain network"
        self.keep_prob = tf.placeholder(dtype=tf.float32 ,name="drop_out")

        self.dw = tf.placeholder(dtype=tf.float32,name = "FlipGradientWeight")

    def creat_model(self):
        self.embed = embed(vocab_size=self.vocab_size,embed_size=self.emb_dim)
        self.cnn_x1 = CNN_model(name ="cnn_x1")         #input tensor: [batch, length, embed_size] kernel_size[fiter_size,len_size]
        self.cnn_x2 = CNN_model(name ="cnn_x2")
        self.att_mat = Attention_3()
        self.fc_layer = FC_model(name= "fc_layer")

    def CNN_layer(self,variable_scope, x1, x2,filter_dim):

        with tf.variable_scope(variable_scope):
            #shared cnn weighted
            L_conv = self.cnn_x1(self.pad_(x1), out_size = self.conv_out, kernel_size=(filter_dim, self.filter_size), pad="VALID", reuse = False)
            # R_conv = self.cnn_x2(x2, kernel_size=(self.filter_size, self.emb_dim),reuse = False)
            R_conv = self.cnn_x1(self.pad_(x2), out_size = self.conv_out, kernel_size=(filter_dim, self.filter_size), pad="VALID", reuse = True)
            L_conv = tf.transpose(L_conv, [0, 3, 2, 1])
            R_conv = tf.transpose(R_conv, [0, 3, 2, 1])

            att_mat = self.att_mat(L_conv, R_conv,variable_name="conv_att")
            L_attention, R_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)
            #pool
            L_wp = w_pool(variable_scope="left", x=L_conv, attention=L_attention, model_type="CNN", length=self.length, filer_size=self.filter_size)
            L_ap = all_pool("left", L_conv, length=self.length, filter_size=self.filter_size,embed_szie=self.emb_dim,conv_out=self.conv_out)
            R_wp = w_pool(variable_scope="right", x=R_conv, attention=R_attention, model_type="CNN", length=self.length, filer_size=self.filter_size)
            R_ap = all_pool("left", R_conv, length=self.length, filter_size=self.filter_size,embed_szie=self.emb_dim,conv_out=self.conv_out)

            return L_wp, L_ap, R_wp, R_ap
    def private_Cnn_layer(self,variavle_scope, x, filter_dim):
        with tf.variable_scope(variavle_scope):
            x = self.cnn_x1(self.pad_(x), out_size = self.conv_out, kernel_size=(filter_dim, self.filter_size),pad="VALID", reuse=False)
            x = slim.avg_pool2d(x,kernel_size=[1,self.length + self.filter_size - 1],stride=1)
            x = tf.reshape(x,[-1,self.conv_out])
            return x
    def input_att(self, x1, x2):
        with tf.variable_scope("att_amt"):
            aW = tf.get_variable(name = "aW", shape=(self.length, self.emb_dim))
            att_mat = self.att_mat(x1, x2)
            x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, aW)), -1)
            x2_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", tf.transpose(att_mat, [0, 2, 1]), aW)), -1)

            # x1 = tf.concat([x1, x1_a], axis = 3)
            # x2 = tf.concat([x2, x2_a], axis = 3)
        return x1_a,x2_a

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

    def compute_loss(self,senti,domain_x1,domain_x2,senti_weight=1.0,domain_weight=1.0):
        senti_loss = tf.losses.softmax_cross_entropy(self.x1_labels, senti) * senti_weight
        domain_loss = (-tf.reduce_mean(domain_x1) + tf.reduce_mean(domain_x2)) * domain_weight
        return senti_loss+domain_loss

    def orthogonality(self,x1, x2, weight=1):
        # def difference_loss(private_samples, shared_samples, weight=1.0, name=''):
        x1 -= tf.reduce_mean(x1, 0)
        x2 -= tf.reduce_mean(x2, 0)
        private_samples = tf.nn.l2_normalize(x1, 1)
        shared_samples = tf.nn.l2_normalize(x2, 1)
        correlation_matrix = tf.matmul( private_samples, shared_samples, transpose_a=True)
        cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
        cost = tf.where(cost > 0, cost, 0, name='value')
        #tf.summary.scalar('losses/Difference Loss {}'.format(name),cost)
        assert_op = tf.Assert(tf.is_finite(cost), [cost])
        with tf.control_dependencies([assert_op]):
            tf.losses.add_loss(cost)
        return cost

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

        s_orthloss = 0
        t_orthloss = 0
        filter_dim = self.emb_dim
        for i in range(self.cnn_layer):
            with tf.device("/gpu:0"), tf.name_scope("convolution_".format(i)):
                if self.use_prob:
                    x1 = tf.nn.dropout(x1,self.keep_prob)
                    x2 = tf.nn.dropout(x2,self.keep_prob)
                L_wp, L_ap, R_wp, R_ap = self.CNN_layer("Cnn_layer_{}".format(i), x1, x2, filter_dim=filter_dim)

        # #jiangdiCNNd tezhenshu
        #     with tf.name_scope("private_feature"):
        #         s_x = self.private_Cnn_layer("source_private_{}".format(i), x1, filter_dim=filter_dim)
        #         t_x = self.private_Cnn_layer("target_private_{}".format(i), x2, filter_dim=filter_dim)
        #         s_orthloss += self.orthogonality(s_x, L_ap)
        #         t_orthloss += self.orthogonality(t_x, R_ap)
        #
        #     x1 = L_wp
        #     x2 = R_wp
        #     filter_dim = self.conv_out

        x1_f = L_ap
        x2_f = R_ap
        if self.use_prob:
            x1_f = tf.nn.dropout(x1_f,self.keep_prob)
            x2_f = tf.nn.dropout(x2_f,self.keep_prob)

        with tf.name_scope("sentiment_analysis"):
            senti_x1 = self.sentiment_fc(x1_f)
            senti_x2 =self.sentiment_fc(x2_f,reuse=True)
        with tf.name_scope("domain_adaption"):
            domain_x1 = self.domain_fc(x1_f)
            domain_x2 = self.domain_fc(x2_f, reuse = True)

        loss = self.compute_loss(senti_x1, domain_x1, domain_x2,senti_weight=0.5,domain_weight=2.0)
        self.loss = loss
        self.loss = loss + s_orthloss + t_orthloss
        self.optimizer(self.opt_method)
        x1_p = self.pred(senti_x1,)
        x2_p = self.pred(senti_x2)

        x1_r = self.pred(self.x1_labels)
        x2_r = self.pred(self.x2_labels)
        self.acc_x1 = self.acc(x1_r, x1_p)
        self.acc_x2 = self.acc(x2_r, x2_p)
        self.Summary()
    def optimizer(self,opt_method = "adam"):
        opt_method = opt_method.lower()
        if opt_method == "adam":
            self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        elif opt_method == "sgd":
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=0.001,).minimize(self.loss)
        elif opt_method == "rms":
            self.opt = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.loss)
        else:
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=0.001,).minimize(self.loss)

    def acc(self,r,p):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(r,p),dtype=tf.float32))
        return accuracy

    def pred(self,x):
        p = tf.argmax(x,axis=1)
        return p

    def Summary(self,):
        tf.summary.scalar("source_acc", self.acc_x1)
        tf.summary.scalar("target_acc", self.acc_x2)
        tf.summary.scalar("loss", self.loss)
        self.summary = tf.summary.merge_all()
        return self.summary

if __name__ == '__main__':
    m = cnn_model(300,100,5000,2,64,32,2000,layer=3)
    m.build()
    pass




