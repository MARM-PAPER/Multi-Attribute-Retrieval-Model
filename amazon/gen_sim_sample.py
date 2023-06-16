import numpy as np
import pandas as pd

RAW_TRAIN_DATA_FILE = '../model_input_amazon/data/book_data/book_train.txt'
RAW_TEST_DATA_FILE = '../model_input_amazon/data/book_data/book_test.txt'

train_output_path = '../model_input_amazon/train_base.pkl'
train_label_path = '../model_input_amazon/train_label.pkl'
test_output_path = '../model_input_amazon/test_base.pkl'
test_label_path = '../model_input_amazon/test_label.pkl'
MAX_LEN_ITEM = 100
def main():
    item_list = []
    cid_list = []
    hist_iid_list = []
    hist_cid_list = []
    label_list = []
    sim_iid_list = []
    sim_cid_list = []
    i = 0
    with open(RAW_TRAIN_DATA_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            uid, iid, cid, label, hist_iid, hist_cid, _, _ = line.strip().split('\t')
            item_list.append(int(iid))
            cid_list.append(int(cid))
            cur_item_list = [int(x) for x in hist_iid.split(',')]
            cur_cat_list = [int(x) for x in hist_cid.split(',')]
            #print("iid", int(iid))
            #print("cid", int(cid))
            #print("cur_item_list", cur_item_list)
            #print("cur_cat_list", cur_cat_list)
            hist_iid_list.append(cur_item_list)
            hist_cid_list.append(cur_cat_list)
            label_list.append(int(label))

            sim_item_list = []
            sim_cat_list = []
            for it in range(len(cur_item_list)):
                if cur_cat_list[it] == int(cid):
                    sim_item_list.append(cur_item_list[it])
                    sim_cat_list.append(cur_cat_list[it])
            sim_item_list = [0] * (MAX_LEN_ITEM - len(sim_item_list)) + sim_item_list    
            sim_cat_list = [0] * (MAX_LEN_ITEM - len(sim_cat_list)) + sim_cat_list 
            #print("sim_item_list", sim_item_list) 
            #print("sim_cat_list", sim_cat_list)
            sim_iid_list.append(sim_item_list)  
            sim_cid_list.append(sim_cat_list)
            #i = i+1
            #if i > 10:
            #    break
             
    pd.to_pickle([np.array(item_list), np.array(cid_list), np.array(hist_iid_list), np.array(hist_cid_list), np.array(sim_iid_list), np.array(sim_cid_list)], train_output_path)
    pd.to_pickle(np.array(label_list), train_label_path)    


    print("test")
    item_list = []
    cid_list = []
    hist_iid_list = []
    hist_cid_list = []
    label_list = []
    sim_iid_list = []
    sim_cid_list = [] 
    with open(RAW_TEST_DATA_FILE, 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            uid, iid, cid, label, hist_iid, hist_cid, _, _ = line.strip().split('\t')
            item_list.append(int(iid))
            cid_list.append(int(cid))
            cur_item_list = [int(x) for x in hist_iid.split(',')]
            cur_cat_list = [int(x) for x in hist_cid.split(',')]
            #print("iid", int(iid))
            #print("cid", int(cid))
            #print("cur_item_list", cur_item_list)
            #print("cur_cat_list", cur_cat_list)
            hist_iid_list.append(cur_item_list)
            hist_cid_list.append(cur_cat_list)
            label_list.append(int(label))

            sim_item_list = []
            sim_cat_list = []
            for it in range(len(cur_item_list)):
                if cur_cat_list[it] == int(cid):
                    sim_item_list.append(cur_item_list[it])
                    sim_cat_list.append(cur_cat_list[it])
            sim_item_list = [0] * (MAX_LEN_ITEM - len(sim_item_list)) + sim_item_list
            sim_cat_list = [0] * (MAX_LEN_ITEM - len(sim_cat_list)) + sim_cat_list
            #print("sim_item_list", sim_item_list)
            #print("sim_cat_list", sim_cat_list)
            sim_iid_list.append(sim_item_list)
            sim_cid_list.append(sim_cat_list)
            #i = i+1
            #if i > 20:
            #    break
            

    pd.to_pickle([np.array(item_list), np.array(cid_list), np.array(hist_iid_list), np.array(hist_cid_list), np.array(sim_iid_list), np.array(sim_cid_list)], test_output_path)
    pd.to_pickle(np.array(label_list), test_label_path)


if __name__ == '__main__':
    main()    

