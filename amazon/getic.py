import _pickle as pkl
import pandas as pd
import random
import numpy as np
import os
import json

RAW_TRAIN_DATA_FILE = '../model_input_amazon/data/book_data/book_train.txt'
RAW_TEST_DATA_FILE = '../model_input_amazon/data/book_data/book_test.txt'

def remap():
    item_cat_dict = {}

    with open(RAW_TRAIN_DATA_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            uid, iid, cid, label, hist_iid, hist_cid, _, _ = line.strip().split('\t')
            item_cat_dict[iid] = cid
            hist_iid_list = hist_iid.split(',')
            hist_cid_list = hist_cid.split(',')
            hist_dict = dict(zip(hist_iid_list, hist_cid_list))
            item_cat_dict.update(hist_dict)

    with open(RAW_TEST_DATA_FILE, 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            uid, iid, cid, label, hist_iid, hist_cid, _, _ = line.strip().split('\t')
            item_cat_dict[iid] = cid
            hist_iid_list = hist_iid.split(',')
            hist_cid_list = hist_cid.split(',')
            hist_dict = dict(zip(hist_iid_list, hist_cid_list))
            item_cat_dict.update(hist_dict)

    res_dict = {'iid': item_cat_dict.keys(), 'cid':item_cat_dict.values()}
    df = pd.DataFrame(res_dict)
    df.to_csv('amazon_item_cat.csv')
    
remap()

