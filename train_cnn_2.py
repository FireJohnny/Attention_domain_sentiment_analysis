#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: train_cnn_2.py
@time: 2018/7/12 18:26
"""


from train_model.model_cnn_2 import *
from data_process.data_process import *
from tensorflow.contrib import learn
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
import pdb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

FLAGS = tf.flags.FLAGS

def parses():
    tf.flags.DEFINE_string("dvd_pos", "./data/dvd/dvdpositive", "dvd Data for the positive data.")
    tf.flags.DEFINE_string("dvd_neg", "./data/dvd/dvdnegative", "dvd Data for the negative data.")
    tf.flags.DEFINE_string("dvd_unl", "./data/English_data/Data/DVDUnlabel.txt", "unlabel dvd data")

    tf.flags.DEFINE_string("elec_pos", "./data/electronics/elecpositive", "electronics Data for the positive data.")
    tf.flags.DEFINE_string("elec_neg", "./data/electronics/elecnegative", "electronics Data for the negative data.")
    tf.flags.DEFINE_string("elec_unl", "./data/English_data/Data/ElectronicsUnlabel.txt", "unlabel elec data")

    tf.flags.DEFINE_string("book_pos", "./data/books/bookpositive", "books Data for the positive data.")
    tf.flags.DEFINE_string("book_neg", "./data/books/booknegative", "books Data for the negative data.")
    tf.flags.DEFINE_string("book_unl", "./data/English_data/Data/BookUnlabel.txt", "unlabel book data")

    tf.flags.DEFINE_string("kitchen_pos", "./data/kitchen/kitchenpositive", "kitchen Data for the positive data.")
    tf.flags.DEFINE_string("kitchen_neg", "./data/kitchen/kitchennegative", "kitchen Data for the negative data.")
    tf.flags.DEFINE_string("kitchen_unl", "./data/English_data/Data/KitchenUnlabel.txt", "unlabel kitchen data")

    tf.flags.DEFINE_integer("emb_size", 300, "Embedding dimensions(default=100)")
    tf.flags.DEFINE_integer("cnn_out", 300, "convolution  out dimensions(default=64)")
    tf.flags.DEFINE_integer("lstm_out", 64, "lstm out dimensions(default=64)")
    tf.flags.DEFINE_integer("cnn_filter_size", 4, "convolution filter size(default=4)")
    tf.flags.DEFINE_integer("cnn_layer", 3, "convolution layers (default=1)")
    tf.flags.DEFINE_string("opt_method", "sgd", "optimizer method:adam,rms,sgd,adadelta (default=sgd)")
    tf.flags.DEFINE_boolean("drop", False, "use dropout method (default=False)")

def data_clean(source,target):
    if source == "dvd":
        s_train, s_labels = load_data_and_labels(FLAGS.dvd_pos, FLAGS.dvd_neg)
        un_s = load_unlabel_data(FLAGS.dvd_unl)
    elif source == "kitchen":
        s_train, s_labels = load_data_and_labels(FLAGS.kitchen_pos, FLAGS.kitchen_neg)
        un_s = load_unlabel_data(FLAGS.kitchen_unl)
    elif source == "elec":
        s_train, s_labels = load_data_and_labels(FLAGS.elec_pos, FLAGS.elec_neg)
        un_s = load_unlabel_data(FLAGS.elec_unl)
    elif source == "book":
        s_train, s_labels = load_data_and_labels(FLAGS.book_pos, FLAGS.book_neg)
        un_s = load_unlabel_data(FLAGS.book_unl)
    else:
        print("源领域输入有错 请重新输入！")
        exit(1)
    if target == "dvd":
        t_train, t_labels = load_data_and_labels(FLAGS.dvd_pos, FLAGS.dvd_neg)
        un_t = load_unlabel_data(FLAGS.dvd_unl)
    elif target == "kitchen":
        t_train, t_labels = load_data_and_labels(FLAGS.kitchen_pos, FLAGS.kitchen_neg)
        un_t = load_unlabel_data(FLAGS.kitchen_unl)
    elif target == "elec":
        t_train, t_labels = load_data_and_labels(FLAGS.elec_pos, FLAGS.elec_neg)
        un_t = load_unlabel_data(FLAGS.elec_unl)
    elif target == "book":
        t_train, t_labels = load_data_and_labels(FLAGS.book_pos, FLAGS.book_neg)
        un_t = load_unlabel_data(FLAGS.book_unl)
    else:
        print("目标领域领域输入有错 请重新输入！")
        exit(1)
    # print("\nParameters:")
    # for key in sorted(FLAGS):
        # v = FLAGS[key].value
        # print("{}={}".format(key,v ))
    print(" ")


    sentence_len = 1000
    tfidf_len = 3000
    sentence_p = learn.preprocessing.VocabularyProcessor(max_document_length=sentence_len)
    sentence_p.fit(un_t + un_s)
    x1_vec = np.array(list(sentence_p.transform(s_train)))
    x2_vec = np.array(list(sentence_p.transform(t_train)))

    tfidf_p = TfidfVectorizer(max_features=tfidf_len)
    tfidf_p.fit(un_t + un_s)
    x1_tf = tfidf_p.transform(s_train).toarray()
    x2_tf = tfidf_p.transform(t_train).toarray()
    shuffle_1 = np.random.permutation(np.arange(len(s_labels)))
    shuffle_2 = np.random.permutation(np.arange(len(t_labels)))
    # pdb.set_trace()
    x1_vec = x1_vec[shuffle_1]
    x2_vec = x2_vec[shuffle_2]
    x1_tf = x1_tf[shuffle_1]
    x2_tf = x2_tf[shuffle_2]
    s_labels = s_labels[shuffle_1]
    t_labels = t_labels[shuffle_2]
    dev_sample_index = -1 * int(0.2 * float(len(x1_vec)))
    x1_train, x1_dev = x1_vec[:dev_sample_index], x1_vec[dev_sample_index:]
    x2_train, x2_dev = x2_vec[:dev_sample_index], x2_vec[dev_sample_index:]
    x1_tf_tr, x1_tf_dev = x1_tf[:dev_sample_index], x1_tf[dev_sample_index:]
    x2_tf_tr, x2_tf_dev = x2_tf[:dev_sample_index], x2_tf[dev_sample_index:]
    x1_labels, x1_dev_labels = s_labels[:dev_sample_index], s_labels[dev_sample_index:]
    x2_labels, x2_dev_labels = t_labels[:dev_sample_index], t_labels[dev_sample_index:]

    data = {
        "x1_train": x1_train,
        "x1_dev": x1_dev,
        "x2_train": x2_train,
        "x2_dev": x2_dev,
        "x1_tf_tr": x1_tf_tr,
        "x1_tf_dev": x1_tf_dev,
        "x2_tf_tr": x2_tf_tr,
        "x2_tf_dev": x2_tf_dev,
        "x1_labels": x1_labels,
        "x1_dev_labels": x1_dev_labels,
        "x2_labels": x2_labels,
        "x2_dev_labels": x2_dev_labels,
        "sentence_len": sentence_len,
        "tfidf_len": tfidf_len,
        "vocab": len(sentence_p.vocabulary_)

    }
    return data

def train(data,checkpoint_dir=None, gpu_id = 0,log_dir = "./log"):



    epoches = 1000
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = cnn_model(emb_dim=FLAGS.emb_size, length=data["sentence_len"],
                              vocab_size=data["vocab"], filter_size=FLAGS.cnn_filter_size,
                              conv_out=FLAGS.cnn_out, lstm_cell=FLAGS.lstm_out,
                              tfidf_len=data["tfidf_len"],layer=FLAGS.cnn_layer,
                              use_prob=FLAGS.drop, opt_method=FLAGS.opt_method)
            model.build()
            saver = tf.train.Saver(tf.global_variables(),max_to_keep=5)
            if checkpoint_dir is None:
                sess.run(tf.global_variables_initializer())
                step =1
                print("init weight sucess")
            else:
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                saver.restore(sess,ckpt.model_checkpoint_path)
                step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                print("load checkponit success")


            if not os.path.exists(log_dir+"/train"):
                os.mkdir(log_dir+"/train")
            if not os.path.exists(log_dir+"/dev"):
                os.mkdir(log_dir+"/dev")
            trainWriter=tf.summary.FileWriter(log_dir+"/train",sess.graph)
            devWriter=tf.summary.FileWriter(log_dir+"/dev",sess.graph)

            # 分布式集成训练，还没学会
            # cluster = tf.train.ClusterSpec()
            # server  = tf.train.Server()

            def train(x1_v, x2_v, x1_tf, x2_tf, x1_label, x2_label, step):
                for b_x1_v, b_x2_v, b_x1_tf, b_x2_tf, b_x1_label, b_x2_label in creat_batch(x1_v, x2_v, x1_tf, x2_tf, x1_label, x2_label, batch_size=10):
                    feed_dict = {model.x1_word: b_x1_v,
                                 model.x2_word: b_x2_v,
                                 model.x1_tf: b_x1_tf,
                                 model.x2_tf: b_x2_tf,
                                 model.x1_labels: b_x1_label,
                                 model.x2_labels: b_x2_label,
                                 model.keep_prob: 0.7,
                                 model.dw:0.1}
                    _, x1_acc, x2_acc, summary = sess.run([model.opt, model.acc_x1, model.acc_x2, model.summary],
                                                     feed_dict=feed_dict)
                    trainWriter.add_summary(summary, global_step=step)
                    print("step:{}, source acc:{}, target acc:{}".format(step, x1_acc, x2_acc))
                    # exit(0)
                    if step > 19999 and step % 100 == 0:
                        # pdb.set_trace()
                        dev_x1_acc,dev_x2_acc = [],[]
                        dev_loss = []
                        for dev_x1, dev_x2, dev_x1_tf, dev_x2_tf, x1_labels, x2_labels,\
                                in creat_batch(data["x1_dev"], data["x2_dev"], data["x1_tf_dev"], data["x2_tf_dev"],
                                               data["x1_dev_labels"], data["x2_dev_labels"],batch_size=10):
                            dev_x1_a,dev_x2_a,d_loss = dev(dev_x1, dev_x2, dev_x1_tf, dev_x2_tf, x1_labels, x2_labels,step)
                            dev_x1_acc.append(dev_x1_a)
                            dev_x2_acc.append(dev_x2_a)
                            dev_loss.append(d_loss)

                        dev_x1_acc = np.mean(dev_x1_acc)
                        dev_x2_acc = np.mean(dev_x2_acc)
                        dev_loss = np.mean(dev_loss)

                        # dev_summary = tf.Summary(value = [tf.Summary.Value(tag="dev_x1_acc",simple_value=dev_x1_acc),
                        #                                   tf.Summary.Value(tag="dev_x2_acc",simple_value=dev_x2_acc),
                        #                                   tf.Summary.Value(tag="dev_loss",simple_value=dev_loss)])

                        dev_summary = tf.Summary()
                        dev_summary.value.add(tag="dev_x1_acc", simple_value=dev_x1_acc)
                        dev_summary.value.add(tag="dev_x2_acc", simple_value=dev_x2_acc)
                        dev_summary.value.add(tag="dev_loss", simple_value=dev_loss)

                        devWriter.add_summary(dev_summary,global_step=step)
                        print("step:{}, source acc:{}, target acc:{}".format(step, dev_x1_acc, dev_x2_acc))
                    if step %1000 == 0:
                        if not os.path.exists(log_dir+"/model"):
                            os.mkdir(log_dir+"/model")
                        saver.save(sess, log_dir+"/model/model", global_step=step)
                    step+=1
                return step
            def dev(dev_x1, dev_x2, dev_x1_tf, dev_x2_tf, x1_labels, x2_labels, step):
                feed_dict = {
                        model.x1_word: dev_x1,
                        model.x2_word: dev_x2,
                        model.x1_tf: dev_x1_tf,
                        model.x2_tf: dev_x2_tf,
                        model.x1_labels: x1_labels,
                        model.x2_labels: x2_labels,
                        model.keep_prob: 1.0
                }
                x1_acc, x2_acc ,d_loss= sess.run([model.acc_x1, model.acc_x2,model.loss], feed_dict=feed_dict)


                return x1_acc,x2_acc,d_loss

            for epoch in range(epoches):
                step = train(data["x1_train"], data["x2_train"], data["x1_tf_tr"], data["x2_tf_tr"], data["x1_labels"], data["x2_labels"], step)
                print("\n\tepoch: {}".format(epoch+1))

    pass


if __name__ == '__main__':
    parses()
    source = sys.argv[1]
    target = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]
    gpu_id = sys.argv[3]
    breakpoint_train = sys.argv[4]

    log_dir = "./log{}".format(gpu_id)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

        os.mkdir(log_dir+"/"+source+"_"+target)
    else:
        os.mkdir(log_dir+"/"+source+"_"+target)
    data = data_clean(source,target)
    if breakpoint_train == "False":
        checkpoint_dir = None
        # print (checkpoint_dir)
    else:
        checkpoint_dir = os.path.join(log_dir, "model")
    train(data,gpu_id=gpu_id,checkpoint_dir=checkpoint_dir,log_dir=log_dir)

    print("source:{} ,target: {}".format(source,target))
    pass



