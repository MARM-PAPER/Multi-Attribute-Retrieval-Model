# coding: utf-8
import numpy as np
import pandas as pd
import os
import json
import sys

MAX_LEN_ITEM=100
np.set_printoptions(threshold=np.inf)
#itemClusters = np.loadtxt('infer.txt')
#itemClusters=pd.read_csv('infer.csv')

item2Clusters = {}
item2Clusters_res = {}
item2Clusters_res_score = {}
#Cluster2item = {}

def item2Cluster(output_path):
    file_names = os.listdir(output_path)
    print('output_path', output_path)
    for file_name in file_names:
        print(file_name)
        with open(output_path+'/'+file_name, 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                data = lines[i].strip().split(',')
                item = data[1]
                clusters = [int(x)+1 for x in data[2:5]]
                clusters_score = [np.float64(x) for x in data[5:8]]
                item2Clusters_res[item] = list(clusters)
                item2Clusters_res_score[item] = list(clusters_score)
                #for cluster in clusters:
                #    if cluster in Cluster2item.keys():
                #        Cluster2item[cluster].add(item)
                #    else:
                #        Cluster2item[cluster] = set([item])
    print("item2Cluster done")

def geneTrainData(input_ori, label, flag, data_tag):
    print(data_tag)
    search_item = []
    search_cat = []
    search_cluster1 = []
    search_cluster2 = []
    search_cluster3 = []
    target_cluster1 = []
    target_cluster2 = []
    target_cluster3 = []
    short_cluster1 = []
    short_cluster2 = []
    short_cluster3 = []    

    train_target_item = input_ori[0]
    train_target_item_cat = input_ori[1]
    
    short_ub_expanded = input_ori[2][:,-30:]
    short_ub_expanded_cat = input_ori[3][:,-30:]
    
    long_ub_expanded = input_ori[2]
    long_ub_expanded_cat = input_ori[3]

    for i in range(len(train_target_item)):
        target_item = str(train_target_item[i])
        seq_item_ori = [str(it) for it in long_ub_expanded[i]]
        seq_cat_ori = long_ub_expanded_cat[i]
        target_clusters = item2Clusters_res[target_item]
        short_ubitem_ori = [str(it) for it in short_ub_expanded[i]]
        short_ubcat_ori = [str(it) for it in short_ub_expanded_cat[i]]
        #print("target_item", target_item)
        #print("target_cate", train_target_item_cat[i])
        #print("target_clusters", target_clusters)
        search_item_c1 = []
        search_item_c2 = []
        search_item_c3 = []
        for j in range(len(seq_item_ori)):
            seq_item = seq_item_ori[j]
            seq_cat = seq_cat_ori[j]
            if seq_item == '0':
                continue
            seq_item_clusters = item2Clusters_res[seq_item]
            #print("seq_item_clusters", seq_item_clusters)
            dict_key = seq_item+'_'+str(seq_cat)+'_'+str(seq_item_clusters[0])+'_'+str(seq_item_clusters[1])+'_'+str(seq_item_clusters[2])
            if target_clusters[0] in seq_item_clusters:
                search_item_c1.append(dict_key)
            elif target_clusters[1] in seq_item_clusters:
                search_item_c2.append(dict_key) 
            elif target_clusters[2] in seq_item_clusters:
                search_item_c3.append(dict_key)
        search_item_c3.extend(search_item_c2)
        search_item_c3.extend(search_item_c1)

        short_cluster1_temp = []
        short_cluster2_temp = []
        short_cluster3_temp = []
        for p in range(len(short_ubitem_ori)):
            short_item = short_ubitem_ori[p]
            if short_item == '0':
                short_cluster1_temp.append(0)
                short_cluster2_temp.append(0)
                short_cluster3_temp.append(0)
            else:
                short_cluster1_temp.append(item2Clusters_res[short_item][0])
                short_cluster2_temp.append(item2Clusters_res[short_item][1])
                short_cluster3_temp.append(item2Clusters_res[short_item][2])
                
        short_cluster1.append(short_cluster1_temp) 
        short_cluster2.append(short_cluster2_temp)
        short_cluster3.append(short_cluster3_temp)

        #print("before---------")
        #for key, value in seq_item_score.items():
        #    print(key, value)
        #print('sorted--------')
        #print(seq_item_score)
        #exit()
        #print("search_item_c1", search_item_c1)
        #print("search_item_c2", search_item_c2)
        #print("search_item_c3", search_item_c3)
        if len(search_item_c3) > MAX_LEN_ITEM:
            search_item_c3=search_item_c3[-MAX_LEN_ITEM:]
        #print("search_item_c3_2", search_item_c3) 
        sim_res = [[np.int64(it.strip().split('_')[0]), np.int64(it.strip().split('_')[1]), np.int64(it.strip().split('_')[2]), np.int64(it.strip().split('_')[3]), np.int64(it.strip().split('_')[4])] for it in search_item_c3]
        #print("sim_res", sim_res)
        sim_res = [[0, 0, 0, 0, 0]] * (MAX_LEN_ITEM - len(sim_res)) + sim_res
        search_item.append(np.array(sim_res)[:,0])
        search_cat.append(np.array(sim_res)[:,1])
        search_cluster1.append(np.array(sim_res)[:,2]) 
        search_cluster2.append(np.array(sim_res)[:,3]) 
        search_cluster3.append(np.array(sim_res)[:,4]) 
        target_cluster1.append(target_clusters[0])
        target_cluster2.append(target_clusters[1])
        target_cluster3.append(target_clusters[2])
    
    print("len:", len(train_target_item))
    output_Path = '../model_input_amazon/' + flag + '_marm_i2c2i_input_' + str(MAX_LEN_ITEM) + data_tag + '.pkl'
    print(output_Path)
    pd.to_pickle([train_target_item, train_target_item_cat, np.array(target_cluster1), np.array(target_cluster2), np.array(target_cluster3), short_ub_expanded, short_ub_expanded_cat, np.array(short_cluster1), np.array(short_cluster2), np.array(short_cluster3), np.array(search_item), np.array(search_cat), np.array(search_cluster1), np.array(search_cluster2), np.array(search_cluster3)], output_Path)

def main():
    output_path = sys.argv[1]
    data_tag = sys.argv[2]
    item2Cluster(output_path)
    train_input_ori = pd.read_pickle('../model_input_amazon/train_base.pkl')
    train_label = pd.read_pickle('../model_input_amazon/train_label.pkl')
    test_input_ori = pd.read_pickle('../model_input_amazon/test_base.pkl')
    test_label = pd.read_pickle('../model_input_amazon/test_label.pkl')
    print('train_label:', len(train_label)) 
    print('test_label:', len(test_label)) 
    geneTrainData(train_input_ori, train_label, "train", data_tag)
    geneTrainData(test_input_ori, test_label, "test", data_tag)


if __name__ == "__main__":
    main()
