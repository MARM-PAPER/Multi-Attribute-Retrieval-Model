import _pickle as pkl
import pandas as pd
import random
import numpy as np
import os
import json

MAX_LEN_ITEM = 256
RAW_DATA_FILE = '../data/taobao_behavior/UserBehavior.csv'

def remap():
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])

    item_key = sorted(df['iid'].unique().tolist())
    item_len = len(item_key)+1
    item_map = dict(zip(item_key, range(1,item_len)))
    df['iid'] = df['iid'].map(lambda x: item_map[x])
    print("iid done")

    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)+1
    cate_map = dict(zip(cate_key, range(1,cate_len)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])
    print("cid done")

    df = df[['iid','cid']].drop_duplicates(inplace=False, ignore_index=True)
    df.to_csv('item_cat.csv')

def to_df():
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
    df = df[['iid','cid']].drop_duplicates(inplace=False, ignore_index=True)
    df.to_csv('item_cat.csv')


remap()

