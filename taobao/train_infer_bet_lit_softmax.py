# coding: utf-8
import os
import sys
import csv
import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from config import RT_SESS_MAX_LEN, LONG_SESS_MAX_LEN, FRAC, TOP_K
import numpy as np
from callback import roc_callback

SESS_MAX_LEN = LONG_SESS_MAX_LEN
long_hist_len_max=LONG_SESS_MAX_LEN
#from models.masm import MASM

gpu_core="3"
if len(sys.argv) > 1:
    gpu_core = sys.argv[1]

model_path = sys.argv[2]
output_path = sys.argv[3] 

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_core
tfconfig = tf.compat.v1.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.compat.v1.Session(config=tfconfig))

if __name__ == "__main__":
    test_input = pd.read_csv('item_cat.csv', header=0, index_col=0)

    print(type(test_input))
    item_arr = test_input['iid'].values
    cate_arr = test_input['cid'].values
    test_size = len(item_arr)

    TEST_BATCH_SIZE = 128

    initial_epoch = 0
    epoch = 1
    
    np.set_printoptions(threshold=np.inf)
    print("model_path", model_path) 
    model = tf.keras.models.load_model(model_path)
    model.summary()
    var_list=tf.compat.v1.global_variables()
    print(var_list)
    #print(model.trainable_variables)
    for v in model.trainable_variables:
        print(v.name, v.shape)
        #if v.name == 'sparse_emb_1-cid/embeddings:0':
        #    print(v.value)
    for v in model.variables:
        print(v.name, v.shape)
    layers=model.layers
    for layer_ in layers:
        print(layer_.name)
    representation_model = Model(inputs=model.inputs, outputs=model.get_layer("tf.nn.softmax").output)
    print("test_size", test_size)
    batch_size = TEST_BATCH_SIZE*10*(SESS_MAX_LEN+1)
    steps = int(test_size/batch_size)
    sample_index = 0 
    for i in range(steps+1):
        if i < steps:
            start = i*batch_size
            end = (i+1)*batch_size
            sample_num = batch_size
            sub_input_item = np.reshape(item_arr[start:end], [TEST_BATCH_SIZE*10, SESS_MAX_LEN+1])
            sub_input_cat = np.reshape(cate_arr[start:end], [TEST_BATCH_SIZE*10, SESS_MAX_LEN+1])
            test_input_sub = [sub_input_item[:,0], sub_input_cat[:,0], sub_input_item[:,1:], sub_input_cat[:, 1:]]
        else:
            start=i*batch_size
            if start > test_size:
                break
            sample_num = int((test_size-start)/(SESS_MAX_LEN+1)) * (SESS_MAX_LEN+1)
            end = start+sample_num
            sub_input_item = np.reshape(item_arr[start:end], [int((test_size-start)/(SESS_MAX_LEN+1)), SESS_MAX_LEN+1])
            sub_input_cat = np.reshape(cate_arr[start:end], [int((test_size-start)/(SESS_MAX_LEN+1)), SESS_MAX_LEN+1])
            test_input_sub = [sub_input_item[:,0], sub_input_cat[:,0], sub_input_item[:,1:], sub_input_cat[:, 1:]]
        sample_index += sample_num 
        print("step:", i)
        print("sample_index", sample_index)
        item2emb = representation_model.predict(test_input_sub, TEST_BATCH_SIZE)
        print(item2emb.shape)
        print(type(item2emb))
        itemid = pd.DataFrame(np.reshape(sub_input_item, [-1, 1]))
        score_arr = np.reshape(item2emb, [sample_num, -1])
        #print("score_arr", score_arr)
        clusters = np.argsort(-1*score_arr, axis=-1)[:,:3]
        clusters_score = -1 * np.sort(-1*score_arr, axis=-1)[:,:3]
        #print("clusters", clusters)
        clusterID = pd.DataFrame(clusters)
        clusterID_score = pd.DataFrame(clusters_score)
        res = pd.concat([itemid, clusterID, clusterID_score], axis=1)
        file_name = output_path + '/infer' + str(i) + '.csv'
        res.to_csv(file_name)

    if sample_index < test_size: 
        print("bef_sample_index", sample_index)
        test_input_sub = [item_arr[sample_index:], cate_arr[sample_index:], np.zeros([(test_size-sample_index), SESS_MAX_LEN]), np.zeros([(test_size-sample_index), SESS_MAX_LEN])]
        sample_index += (test_size-sample_index) 
        print("sample_index", sample_index)
        print("test_input_sub_size", len(test_input_sub[0]))
        item2emb = representation_model.predict(test_input_sub, TEST_BATCH_SIZE)
        print(item2emb.shape)
        print(type(item2emb))
        itemid = pd.DataFrame(np.reshape(test_input_sub[0][:test_size], [-1, 1]))
        score_arr = item2emb[:,0,:]
        clusters = np.argsort(-1*score_arr, axis=-1)[:,:3]
        clusters_score = -1 * np.sort(-1*score_arr, axis=-1)[:,:3]
        clusterID = pd.DataFrame(clusters)
        clusterID_score = pd.DataFrame(clusters_score)
        res = pd.concat([itemid, clusterID, clusterID_score], axis=1)
        file_name = output_path + '/infer' + str(i+1) + '.csv'
        res.to_csv(file_name)
