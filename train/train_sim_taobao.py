# coding: utf-8
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K
from deepctr.utils import SingleFeat
from config import DIN_SESS_MAX_LEN, FRAC, SIM_SESS_MAX_100LEN
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
    SESS_MAX_LEN = DIN_SESS_MAX_LEN
    fd = pd.read_pickle('../model_input_taobao/sim_fd_256v3.pkl')
    fd_hist = pd.read_pickle('../model_input_taobao/sim_fd_256v3.pkl')
    fd['sparse']=fd['sparse'][1:3]
    fd_hist['sparse']=fd_hist['sparse'][1:3]
    print(type(fd))
    train_input_ori = pd.read_pickle('../model_input_taobao/train_sim_input_256v3.pkl')
    train_label = pd.read_pickle('../model_input_taobao/train_sim_label_256v3.pkl')

    test_input_ori = pd.read_pickle('../model_input_taobao/test_sim_input_256v3.pkl')
    test_label = pd.read_pickle('../model_input_taobao/test_sim_label_256v3.pkl')

    ins_num = len(train_input_ori[1])
    print("ins_num", ins_num)

    train_input = [train_input_ori[1], train_input_ori[2], train_input_ori[3][:,-1*DIN_SESS_MAX_LEN:], train_input_ori[4][:,-1*DIN_SESS_MAX_LEN:], train_input_ori[5][:,-1*SIM_SESS_MAX_100LEN:], train_input_ori[6][:,-1*SIM_SESS_MAX_100LEN:]]
    print(train_input)
    print(train_label)

    test_size = int(0.5 * len(test_input_ori[0]))
    print("test_size:", test_size)
    test_input = [test_input_ori[1][:test_size],test_input_ori[2][:test_size], test_input_ori[3][:test_size,-1*DIN_SESS_MAX_LEN:], test_input_ori[4][:test_size,-1*DIN_SESS_MAX_LEN:], test_input_ori[5][:test_size,-1*SIM_SESS_MAX_100LEN:], test_input_ori[6][:test_size,-1*SIM_SESS_MAX_100LEN:]]
    test_label = test_label[:test_size]
    print("test info")
    sess_len_max = SESS_MAX_LEN
    BATCH_SIZE = 256

    sess_feature = ['iid', 'cid']
    TEST_BATCH_SIZE = 1024

    print('fd', fd)
    model = SIM_LN_SOFTMAX01_V3(fd, fd_hist, sess_feature, embedding_size=16, att_activation='dice',
                att_weight_normalization=False, hist_len_max=sess_len_max, dnn_hidden_units=(200, 80),
                att_hidden_size=(80, 40,), hist_long_len_max=SIM_SESS_MAX_100LEN)

    #model.compile('adagrad', 'binary_crossentropy',
    #              metrics=['binary_crossentropy', ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['binary_crossentropy', ])

    hist_ = model.fit(train_input[:], train_label,
                      batch_size=BATCH_SIZE, epochs=1, initial_epoch=0, verbose=1, 
                      callbacks=[roc_callback(validation_data=(test_input, test_label))])
    pred_ans = model.predict(test_input, TEST_BATCH_SIZE)

    print("test LogLoss", round(log_loss(test_label, pred_ans), 4), "test AUC",
          round(roc_auc_score(test_label, pred_ans), 4))
