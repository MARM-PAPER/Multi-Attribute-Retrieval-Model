# coding: utf-8
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K
from deepctr.utils import SingleFeat
from config import AMAZON_SHORT_SESS_MAX_LEN, FRAC, AMAZON_SIM_SESS_MAX_LEN
from models import SIM_LN_SOFTMAX01_V3
import numpy as np
import sys
from callback import roc_callback

gpu_core="3"
gpu_core = sys.argv[1]
data_tag = sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_core

print(data_tag)

tfconfig = tf.compat.v1.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.compat.v1.Session(config=tfconfig))

if __name__ == "__main__":
    FRAC = FRAC
    fd = {}
    fd['sparse'] = [SingleFeat('iid', 450000), SingleFeat('cid', 450000)]
    fd['sparse'].append(SingleFeat('clu1', 33))
    fd['sparse'].append(SingleFeat('clu2', 33))
    fd['sparse'].append(SingleFeat('clu3', 33))
    print(type(fd))
    print(fd)

    train_input_ori = pd.read_pickle("../model_input_amazon/train_marm_i2c2i_input_100" + data_tag + ".pkl")
    train_label = pd.read_pickle('../model_input_amazon/train_label.pkl')

    test_input_ori = pd.read_pickle('../model_input_amazon/test_marm_i2c2i_input_100'+ data_tag + '.pkl')
    test_label = pd.read_pickle('../model_input_amazon/test_label.pkl')

    ins_num = len(train_input_ori[1])
    print("ins_num", ins_num)

    train_input = [train_input_ori[0], train_input_ori[1], train_input_ori[2], train_input_ori[3], train_input_ori[4], train_input_ori[5][:,-1*AMAZON_SHORT_SESS_MAX_LEN:], train_input_ori[6][:,-1*AMAZON_SHORT_SESS_MAX_LEN:], train_input_ori[7][:,-1*AMAZON_SHORT_SESS_MAX_LEN:], train_input_ori[8][:,-1*AMAZON_SHORT_SESS_MAX_LEN:], train_input_ori[9][:,-1*AMAZON_SHORT_SESS_MAX_LEN:], train_input_ori[10][:,-1*AMAZON_SIM_SESS_MAX_LEN:], train_input_ori[11][:,-1*AMAZON_SIM_SESS_MAX_LEN:], train_input_ori[12][:,-1*AMAZON_SIM_SESS_MAX_LEN:], train_input_ori[13][:,-1*AMAZON_SIM_SESS_MAX_LEN:], train_input_ori[14][:,-1*AMAZON_SIM_SESS_MAX_LEN:]]
    print(train_input)
    print(train_label)

    test_size = len(test_input_ori[0])
    print("test_size:", test_size)
    test_input = [test_input_ori[0][:test_size],test_input_ori[1][:test_size], test_input_ori[2][:test_size], test_input_ori[3][:test_size], test_input_ori[4][:test_size], test_input_ori[5][:test_size,-1*AMAZON_SHORT_SESS_MAX_LEN:], test_input_ori[6][:test_size,-1*AMAZON_SHORT_SESS_MAX_LEN:], test_input_ori[7][:test_size,-1*AMAZON_SHORT_SESS_MAX_LEN:], test_input_ori[8][:test_size,-1*AMAZON_SHORT_SESS_MAX_LEN:], test_input_ori[9][:test_size,-1*AMAZON_SHORT_SESS_MAX_LEN:],test_input_ori[10][:test_size,-1*AMAZON_SIM_SESS_MAX_LEN:], test_input_ori[11][:test_size,-1*AMAZON_SIM_SESS_MAX_LEN:], test_input_ori[12][:test_size,-1*AMAZON_SIM_SESS_MAX_LEN:], test_input_ori[13][:test_size,-1*AMAZON_SIM_SESS_MAX_LEN:], test_input_ori[14][:test_size,-1*AMAZON_SIM_SESS_MAX_LEN:]]
    test_label = test_label[:test_size]
    print("test info")
    BATCH_SIZE = 128

    sess_feature = ['iid', 'cid', 'clu1', 'clu2' ,'clu3']
    TEST_BATCH_SIZE = 1024

    print('fd', fd)
    model = SIM_LN_SOFTMAX01_V3(fd, fd, sess_feature, embedding_size=16, att_activation='dice',
                att_weight_normalization=False, hist_len_max=AMAZON_SHORT_SESS_MAX_LEN, dnn_hidden_units=(200, 80),
                att_hidden_size=(80, 40,), hist_long_len_max=AMAZON_SIM_SESS_MAX_LEN, temp=2.3)

    #model.compile('adagrad', 'binary_crossentropy',
    #              metrics=['binary_crossentropy', ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['binary_crossentropy', ])

    hist_ = model.fit(train_input[:], train_label,
                      batch_size=BATCH_SIZE, epochs=1, initial_epoch=0, verbose=1, 
                      callbacks=[roc_callback(validation_data=(test_input, test_label))])
    pred_ans = model.predict(test_input, TEST_BATCH_SIZE)

    print("test LogLoss", round(log_loss(test_label, pred_ans), 4), "test AUC",
          round(roc_auc_score(test_label, pred_ans), 4))
