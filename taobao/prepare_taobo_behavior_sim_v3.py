#import cPickle as pkl
import _pickle as pkl
import pandas as pd
import random
import numpy as np
from deepctr.utils import SingleFeat
import os
import json

RAW_DATA_FILE = '../data/taobao_behavior/UserBehavior.csv'

MAX_LEN_ITEM = 256
data_tag='v3'
log_path="./sim_te.log"

def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
    return df

def remap(df):
    item_key = sorted(df['iid'].unique().tolist())
    item_len = len(item_key)+1
    item_map = dict(zip(item_key, range(1,item_len)))
    df['iid'] = df['iid'].map(lambda x: item_map[x])
    print("iid done")

    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)+1
    user_map = dict(zip(user_key, range(1,user_len)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])
    print("uid done")

    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)+1
    cate_map = dict(zip(cate_key, range(1,cate_len)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])
    print("cid done")

    btag_key = sorted(df['btag'].unique().tolist())
    btag_len = len(btag_key)+1
    btag_map = dict(zip(btag_key, range(1,btag_len)))
    df['btag'] = df['btag'].map(lambda x: btag_map[x])
    print('batg done')

    print(item_len, user_len, cate_len, btag_len)
    return df, item_len, user_len, cate_len, user_len + item_len + cate_len + btag_len + 1 #+1 is for unknown target btag


def gen_user_item_group(df, cate_cnt, feature_size):
    print("begin group")
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    cate_df = df.sort_values(['cid', 'time']).groupby('cid')
    index = 0
    cid_dict = {}
    for ccid, hist in cate_df:
        cid_dict[ccid] = index
        index += 1
    print("group completed")
    return user_df, cate_df, cid_dict

def gen_dataset(user_df, cate_df, cate_cnt, feature_size, cid_dict):
    print("begin gen_dataset")
    train_sample_list = []
    test_sample_list = []
    # get each user's last touch point time
    cate_df_list = list(cate_df) 
    print(len(user_df))

    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])
    print("get user last touch time completed")

    user_last_touch_time_sorted = sorted(user_last_touch_time)
    print("sort ok")
    split_time = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.8)]
    print("split_time ok")

    cnt = 0
    noneg_num = 0
    for uid, hist in user_df:
        cnt += 1
        if (cnt % 10000 == 0):
            print(cnt)
        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        btag_hist = hist['btag'].tolist()
        target_item_time = hist['time'].tolist()[-1]

        target_item = item_hist[-1]
        target_item_cate = cate_hist[-1]
        target_item_btag = feature_size
        label = 1
        test = (target_item_time > split_time)
        # neg sampling
        neg = random.randint(0, 1)
        if neg == 1:
            label = 0

            cur_cate = target_item_cate
            cur_cat_item_list = cate_df_list[cid_dict[cur_cate]][1]['iid'].unique().tolist()
            cur_item_len = len(cur_cat_item_list)
            if cur_item_len == 1:
                noneg_num += 1
                continue
            while target_item == item_hist[-1]:
                sample_index = random.randint(0, cur_item_len - 1)
                target_item = cur_cat_item_list[sample_index]
                target_item_btag = feature_size
        # the item history part of the sample
        item_part = []
        item_part_sim = []
        for i in range(len(item_hist) - 1):
            if cate_hist[i] == target_item_cate:
                item_part_sim.append([uid, item_hist[i], cate_hist[i], btag_hist[i]])
            item_part.append([uid, item_hist[i], cate_hist[i], btag_hist[i]])
        #item_part.append([uid, target_item, target_item_cate, target_item_btag])
        #item_part_sim.append([uid, target_item, target_item_cate, target_item_btag])
        # item_part_len = min(len(item_part), MAX_LEN_ITEM)

        # choose the item side information: which user has clicked the target item
        # padding history with 0
        if len(item_part) <= MAX_LEN_ITEM:
            item_part_pad =  [[0] * 4] * (MAX_LEN_ITEM - len(item_part)) + item_part
        else:
            item_part_pad = item_part[len(item_part) - MAX_LEN_ITEM:len(item_part)]
        
        if len(item_part_sim) <= MAX_LEN_ITEM:
            item_part_pad_sim =  [[0] * 4] * (MAX_LEN_ITEM - len(item_part_sim)) + item_part_sim
        else:
            item_part_pad_sim = item_part_sim[len(item_part_sim) - MAX_LEN_ITEM:len(item_part_sim)]
        # gen sample
        # sample = (label, item_part_pad, item_part_len, user_part_pad, user_part_len)
        cat_list = []
        item_list = []
        cat_list_sim = []
        item_list_sim = []
        if test:
            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
            for i in range(len(item_part_pad_sim)):
                item_list_sim.append(item_part_pad_sim[i][1])
                cat_list_sim.append(item_part_pad_sim[i][2])

            test_sample_list.append([uid, target_item, target_item_cate, label, item_list, cat_list, item_list_sim, cat_list_sim])
        else:
            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
            for i in range(len(item_part_pad_sim)):
                item_list_sim.append(item_part_pad_sim[i][1])
                cat_list_sim.append(item_part_pad_sim[i][2])
            train_sample_list.append([uid, target_item, target_item_cate, label, item_list, cat_list, item_list_sim, cat_list_sim])
    
    print("total_num:", cnt)
    print("noneg_num:", noneg_num)
    train_sample_length_quant = int(len(train_sample_list)/256*256)
    test_sample_length_quant = int(len(test_sample_list)/256*256)
    train_sample_list = train_sample_list[:train_sample_length_quant]
    test_sample_list = test_sample_list[:test_sample_length_quant]
    random.shuffle(train_sample_list)
    print("length",len(train_sample_list))
    return train_sample_list, test_sample_list


def produce_neg_item_hist_with_cate(train_file, test_file):
    write_pkl(train_file, "train")
    write_pkl(test_file, "test")    

def write_pkl(sample_data, flag):
    uid_list = []
    iid_list = []
    cid_list = []
    label_list = []
    his_iid_list = []
    his_cid_list = []
    neg_his_iid_list = []
    neg_his_cid_list = []
    for sample in sample_data:
        uid_list.append(sample[0])
        iid_list.append(sample[1])
        cid_list.append(sample[2])
        label_list.append(sample[3])
        his_iid_list.append(sample[4])
        his_cid_list.append(sample[5])
        neg_his_iid_list.append(sample[6])
        neg_his_cid_list.append(sample[7])
    
    pd.to_pickle([np.array(uid_list), np.array(iid_list), np.array(cid_list), np.array(his_iid_list), np.array(his_cid_list), np.array(neg_his_iid_list), np.array(neg_his_cid_list)], '../model_input_taobao/' + flag + '_sim_input_' + str(MAX_LEN_ITEM) + data_tag + '.pkl') 
    pd.to_pickle(np.array(label_list), '../model_input_taobao/' + flag + '_sim_label_' + str(MAX_LEN_ITEM) + data_tag + '.pkl') 
    
def write_log(logging):
    print(logging, log_path)
def main():

    if not os.path.exists('../model_input_taobao/'):
        os.mkdir('../model_input_taobao/')
    df = to_df(RAW_DATA_FILE)
    print("to_df")
    df, item_len, user_len, cate_len, feature_size = remap(df)
    print("remap")
    sparse_feature_list = [SingleFeat('uid', user_len), SingleFeat('iid', item_len), SingleFeat('cid', cate_len)]
    pd.to_pickle({'sparse': sparse_feature_list},'../model_input_taobao/sim_fd_' + str(MAX_LEN_ITEM) + data_tag + '.pkl')
    print("save sim_fd succ")


    user_df, cate_df, cid_dict = gen_user_item_group(df, cate_len, feature_size)
    print("gen_user_item_group")
    train_sample_list, test_sample_list = gen_dataset(user_df, cate_df, cate_len, feature_size, cid_dict)
    print("gen_dataset")
    produce_neg_item_hist_with_cate(train_sample_list, test_sample_list)
    print("produce_neg_item_hist_with_cate")


if __name__ == '__main__':
    main()
