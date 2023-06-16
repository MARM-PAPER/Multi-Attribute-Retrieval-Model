# coding: utf-8
import os
import sys

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K

from config import RT_SESS_MAX_LEN, LONG_SESS_MAX_LEN, FRAC, TOP_K


from callback import roc_callback


SESS_MAX_LEN = LONG_SESS_MAX_LEN
long_hist_len_max=LONG_SESS_MAX_LEN
#from models.masm import MASM
if len(sys.argv) > 1:
    if sys.argv[1] == "AADN":
        print('AADN')
        from models.Pre_AADN import AADN

gpu_core="3"
if len(sys.argv) > 2:
    gpu_core = sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_core

lr = 0.001
BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
attr_num = 256
debug = False
contra_temperature=0.5
contra_loss_weight=0.01
contra_dropout_rate = 0.2
gate_temperature = 0.75
params = {}
if len(sys.argv) > 3 and '=' in sys.argv[3]:
    info = dict([item.split('=') for item in sys.argv[3].split(',')])
    if 'batch_size' in info:
        BATCH_SIZE = int(info['batch_size'])
        TEST_BATCH_SIZE = int(info['batch_size'])
    if 'attr_num' in info:
        attr_num = int(info['attr_num'])
    if 'expert_num' in info:
        expert_num = int(info['expert_num'])
    if 'debug' in info:
        debug = bool(info['debug'])
    if 'contra_temperature' in info:
        contra_temperature = float(info['contra_temperature'])
    if 'contra_loss_weight' in info:
        contra_loss_weight = float(info['contra_loss_weight'])
    if 'contra_dropout_rate' in info:
        contra_dropout_rate = float(info['contra_dropout_rate'])
    if 'gate_temperature' in info:
        gate_temperature = float(info['gate_temperature'])

params['batch_size'] = BATCH_SIZE
params['attr_num'] = attr_num
params['debug'] = debug
params['contra_temperature'] = contra_temperature
params['contra_loss_weight'] = contra_loss_weight
params['contra_dropout_rate'] = contra_dropout_rate
params['gate_temperature'] = gate_temperature
print(params)


checkpoint_dir = "ckpt/" + sys.argv[1] + "_masm" + sys.argv[3]

tfconfig = tf.compat.v1.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.compat.v1.Session(config=tfconfig))

if __name__ == "__main__":
    fd = pd.read_pickle('../model_input_taobao/sim_fd_256v3.pkl')
    fd['sparse'] = fd['sparse'][1:]
    print(type(fd))
    train_input = pd.read_pickle('../model_input_taobao/train_sim_input_256v3.pkl')
    train_label = pd.read_pickle('../model_input_taobao/train_sim_label_256v3.pkl')

    test_input = pd.read_pickle('../model_input_taobao/test_sim_input_256v3.pkl')
    test_label = pd.read_pickle('../model_input_taobao/test_sim_label_256v3.pkl')

    train_input = [train_input[1],train_input[2], train_input[3][:,-1*SESS_MAX_LEN:], train_input[4][:,-1*SESS_MAX_LEN:]]
    print("train_input", train_input)
    print("train_label", train_label)

    test_size = int(0.5 * len(test_input[0]))
    test_input = [test_input[1][:test_size],test_input[2][:test_size], test_input[3][:test_size, -1*SESS_MAX_LEN:], test_input[4][:test_size, -1*SESS_MAX_LEN:]]
    test_label = test_label[:test_size]
    print("test_input", test_input)
    print("test_label", test_label)

    params['long_hist_len_max'] = long_hist_len_max

    sess_feature = ['iid', 'cid']

    initial_epoch = 0
    epoch = 1

    print('fd', fd)
    model = AADN(fd, sess_feature, embedding_size=16, att_activation='dice', init_std=0.01,
                att_weight_normalization=False, rt_hist_len_max=RT_SESS_MAX_LEN, long_hist_len_max=long_hist_len_max, dnn_hidden_units=(200, 80),
                att_hidden_size=(80, 40,), topk=TOP_K, params=params)


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['binary_crossentropy', ])

    hist_ = model.fit(train_input[:], train_label,
                      #steps_per_epoch=1000,
                      batch_size=BATCH_SIZE, epochs=epoch, initial_epoch=initial_epoch, verbose=1,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + '/model_{epoch:02d}', save_weights_only=False)]
                )#, callbacks=[roc_callback(validation_data=(test_input, test_label))])
    pred_ans = model.predict(test_input, TEST_BATCH_SIZE)

    print("test: LogLoss_main", round(log_loss(test_label, pred_ans), 4),
            "AUC_main", round(roc_auc_score(test_label, pred_ans), 4))
