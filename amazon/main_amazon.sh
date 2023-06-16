#!/bin/bash
set -eux

tar -jxvf data.tar.bz2
mkdir ../model_input_amazon
mv data ../model_input_amazon

#general data used by din&sim
python gen_sim_sample.py

#general all item_cate
python getic.py

gpu_core=3
save_model_name="test"
infer_output_path="infer_res"
data_tag="test"
mkdir $infer_output_path

#train AADN
python ../train/train_pretrain_amazon.py AADN ${gpu_core} ${save_model_name}

#general marm data
model_path_savedBy_AADN=''
python train_infer_bet_lit_softmax.py $gpu_core ${model_path_savedBy_AADN} $infer_output_path
python -u i2c2i_v2_side_seq.py $infer_output_path $data_tag

#train MARM
python ../train/train_marm_amazon.py $gpu_core $data_tag
