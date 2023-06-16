# -*- coding:utf-8 -*-

from collections import OrderedDict

import tensorflow as tf
from deepctr.input_embedding import get_inputs_list, create_singlefeat_inputdict, get_embedding_vec_list
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.sequence import AttentionSequencePoolingLayer
from deepctr.layers.utils import concat_fun, NoMask
from deepctr.utils import check_feature_config_dict
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Dropout, Input, Dense, Embedding, Concatenate, Flatten, Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras import backend as K
import numpy as np

debug_info = []
debug_info_2 = []

class matmulLayer(Layer):
    def __init__(self, attr_num):
        super(matmulLayer, self).__init__()
        self.attr_num = attr_num
        self.attr_embedding =  self.add_weight(name="attr_embedding", shape=[attr_num, attr_num], initializer=tf.keras.initializers.orthogonal(), trainable=True)

    def call(self, inputs):
        attr_embedding = tf.nn.l2_normalize(self.attr_embedding, axis=1)
        return tf.matmul(inputs, attr_embedding), attr_embedding
        #return tf.matmul(inputs, tf.stop_gradient(self.attr_embedding)), self.attr_embedding


def get_input(feature_dim_dict, seq_feature_list, long_seq_max_len):
    sparse_input, dense_input = create_singlefeat_inputdict(feature_dim_dict)
    user_behavior_input = OrderedDict()
    for i, feat in enumerate(seq_feature_list):
        user_behavior_input[feat] = Input(shape=(long_seq_max_len,), name='seq_' + str(i) + '-' + feat)

    return sparse_input, dense_input, user_behavior_input

def attr_util_loss(attr_scores, params):
    attr_num = params['attr_num']
    # [-1, 1 + seq_length, attr_num]
    attr_scores = tf.transpose(tf.reshape(attr_scores, [-1, attr_num]), [1, 0])  # [attr_num, -1]
    print('attr util loss, attr_scores', attr_scores)
    attr_util = tf.reduce_sum(attr_scores, axis=-1)  # [attr_num, 1]
    util_loss = 0.001 * tf.math.reduce_std(attr_util)
    debug_info.append(['util loss', util_loss])
    return util_loss


def contra_loss(attr_embedding, params):
    # attr_embedding: [dim, attr_num]
    attr_num = params['attr_num']
    seqs_length = params['long_hist_len_max']
    contra_t = params['contra_temperature']
    loss_weight = params['contra_loss_weight']
    dropout_rate = params['contra_dropout_rate']

    # augmentation
    attr_embedding_aug = Dropout(dropout_rate)(attr_embedding)
    
    # pos_pair
    debug_info.append(['before dropout', attr_embedding[:2,:]])
    debug_info.append(['after dropout', attr_embedding_aug[:2,:]])
    pos = tf.math.exp(tf.reduce_sum(tf.multiply(attr_embedding, attr_embedding_aug), axis=1, keepdims=True) / contra_t)
    debug_info.append(['pos', pos])
    print('pos', pos, tf.multiply(attr_embedding, attr_embedding_aug))
    neg_pair = tf.math.exp(tf.matmul(attr_embedding, tf.transpose(attr_embedding)) / contra_t)
    neg = tf.reduce_sum(neg_pair - tf.linalg.diag(tf.linalg.diag_part(neg_pair)), axis=1, keepdims=True)
    print('neg', neg, tf.matmul(attr_embedding, tf.transpose(attr_embedding)))
    debug_info.append(['neg', neg])
    loss = -1 * loss_weight * tf.reduce_mean(tf.math.log(pos / (pos + neg)))
    print('loss', loss, tf.math.log(pos / (pos + neg)))
    debug_info.append(['contra_loss', loss])
    return loss
    
def gumber_softmax(x, tau, shape):
    # noise = np.random.gumbel(size=shape)
    # noisy_x = x + noise
    # x = tf.nn.softmax(noisy_x / tau, -1)
    eps = 0.00000001
    uniform_dist = tf.random.uniform(tf.shape(x), 0, 1)
    gumbel_dist = -1 * tf.math.log(
        -1 * tf.math.log(uniform_dist + eps) + eps
    )
    noisy_x = x + gumbel_dist
    x = tf.nn.softmax(noisy_x / tau, -1)
    return x


def attr_project_model(attr_input, params):
    attr_num = params['attr_num']
    gate_temperature = params['gate_temperature']
    fc1 = DNN(hidden_units=[256, 128], activation='dice', dropout_rate=0, l2_reg=0, use_bn=True, seed=1024)(
        attr_input)
    fc2 = DNN(hidden_units=[attr_num], activation='linear', dropout_rate=0, l2_reg=0, use_bn=True,
              seed=1024)(fc1)
    print(fc2)

    attr_scores = tf.keras.layers.LayerNormalization(axis=-1)(fc2)

    shape = [params['batch_size'], params['long_hist_len_max'] + 1, attr_num]
    attr_scores = gumber_softmax(attr_scores, gate_temperature, shape)
    debug_info_2.append(['attr_scores_bef', attr_scores])
    attr_express, attr_embedding = matmulLayer(attr_num)(attr_scores)
    #print('attr_embedding2', attr_embedding)
    debug_info_2.append(['attr_express', attr_express])
    debug_info_2.append(['attr_embedding', attr_embedding])
    #
    return attr_scores, attr_express, attr_embedding, attr_num


def main_tower(attr_express, attr_num, dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed,
               task):
    # mlp
    target_attr = tf.reshape(attr_express[:, 0, :], [-1, 1, attr_num])
    seqs_attr = attr_express[:, 1:, :]

    # output layer
    attention_score = tf.matmul(target_attr, seqs_attr, transpose_b=True)
    print('attention_score', attention_score)
    attention_score = tf.transpose(tf.tile(attention_score, [1, attr_num, 1]), [0, 2, 1])
    hist_attr = seqs_attr * attention_score
    hist_attr = tf.reduce_sum(hist_attr, axis=1)
    print('hist_attr', hist_attr)

    target_attr = tf.reshape(target_attr, [-1, attr_num])
    deep_input = tf.concat([target_attr, hist_attr, target_attr - hist_attr, target_attr * hist_attr], axis=-1)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, dnn_use_bn, seed)(deep_input)
    final_logit = Dense(1, use_bias=True)(output)
    output = PredictionLayer(task)(final_logit)
    return output


def AADN(feature_dim_dict, seq_feature_list, embedding_size=8, attr_embedding_size=8,  rt_hist_len_max=100, long_hist_len_max=400,
         dnn_use_bn=False, dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40),
         att_activation="dice", att_weight_normalization=False, topk=50, params={},
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary'):
    """Instantiates the Deep Interest Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field (**now only support sparse feature**)like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':[]}
    :param seq_feature_list: list,to indicate  sequence sparse field (**now only support sparse feature**),must be a subset of ``feature_dim_dict["sparse"]``
    :param embedding_size: positive integer,sparse feature embedding_size.
    :param hist_len_max: positive int, to indicate the max length of seq input
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """
    # fd {'sparse': [SingleFeat(name='uid', dimension=987994, hash_flag=False, dtype='float32'), SingleFeat(name='iid', dimension=4162024, hash_flag=False, dtype='float32'), SingleFeat(name='cid', dimension=9439, hash_flag=False, dtype='float32')]}
    # seq_feature_list = ['iid', 'cid']
    check_feature_config_dict(feature_dim_dict)

    sparse_input, dense_input, user_behavior_input = get_input(
        feature_dim_dict, seq_feature_list, long_hist_len_max)

    print("sparse_input", sparse_input)
    print("dense_input", dense_input)

    # ********************* embedding layer *****************************************
    sparse_embedding_dict = {feat.name: Embedding(feat.dimension, embedding_size,
                                                  embeddings_initializer=RandomNormal(
                                                      mean=0.0, stddev=init_std, seed=seed),
                                                  embeddings_regularizer=l2(
                                                      l2_reg_embedding),
                                                  name='sparse_emb_' + str(i) + '-' + feat.name,
                                                  mask_zero=(feat.name in seq_feature_list)) for i, feat in
                             enumerate(feature_dim_dict["sparse"])}

    print('sparse_embedding_dict',
          {str(i) + " " + feat.name: feat.dimension for i, feat in enumerate(feature_dim_dict["sparse"])})

    # **********************masm************************************************
    # target cid, iid
    target_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'],
                                             seq_feature_list, seq_feature_list)
    # seqs cid, iid
    long_seqs_input = OrderedDict()
    for feat in user_behavior_input:
        long_seqs_input[feat] = user_behavior_input[feat][:, -long_hist_len_max:]
    seqs_emb_list = get_embedding_vec_list(sparse_embedding_dict, long_seqs_input, feature_dim_dict['sparse'],
                                           seq_feature_list, seq_feature_list)
    # target emb, seqs emb
    target_emb = sum(target_emb_list)
    seqs_emb = sum(seqs_emb_list)
    print('target_emb_list', target_emb_list)
    print('target_emb', target_emb)
    print("seqs_emb_list", seqs_emb_list)
    print('seqs_emb', seqs_emb)

    # attr projection layer
    attr_input = tf.concat([target_emb, seqs_emb], axis=1)
    attr_scores, attr_express, attr_embedding, attr_num = attr_project_model(attr_input, params)

    # for debug
    topk_score, topk_attr = tf.math.top_k(attr_scores[:5, 0, :], k=5)
    debug_info.append(["topk_score", topk_score])
    debug_info.append(["topk_attr", topk_attr])

    # main tower
    output = main_tower(attr_express, attr_num, dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                        seed, task)

    # hard search
    aux_loss_1 = contra_loss(attr_embedding, params)
    aux_loss_2 = attr_util_loss(attr_scores, params)
    aux_loss = aux_loss_1 + aux_loss_2

    # debug
    if params['debug']:
        output = tf.keras.backend.print_tensor(output, debug_info)
    # tf.compat.v1.Print(output, debug_info, summarize=-1)
    model_input_list = get_inputs_list([sparse_input, dense_input, user_behavior_input])

    model = Model(inputs=model_input_list, outputs=output)

    model.add_loss(aux_loss)
    return model
