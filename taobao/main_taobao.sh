#!/bin/bash
set -eux

#general data used by din&sim
python prepare_taobo_behavior_sim_v3.py

#general all item_cate
python getic.py

#train AADN
python ../train/train_pretrain_taobao.py AADN ${gpu_core} ${save_model_name} 

#general marm data
python train_infer_bet_lit_softmax.py $gpu_core ${model_path_savedBy_AADN} $infer_output_path
python i2c2i_v2_side_seq.py $infer_output_path $data_tag

#train MARM
python ../train/train_marm_taobao.py $gpu_core $data_tag
