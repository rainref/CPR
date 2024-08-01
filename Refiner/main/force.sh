#!/bin/bash
var=0
ocp_memory=${2:-500}
while [ $var -eq 0 ]
echo 'waiting for available gpu...'
do
    count=0
    for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    do
        if [ $i -lt $ocp_memory ]
        then
            echo 'GPU'$count' is avaiable'
                        python3 train.py --data_workers 5 --model_dir java_base --model_name java_block --uncase False --use_src_word True --use_src_char False --use_tgt_word True --use_tgt_char False --max_src_len 420 --max_tgt_len 80 --emsize 512 --fix_embeddings False --src_vocab_size 80000 --share_decoder_embeddings True --max_examples -1 --batch_size 138 --test_batch_size 256 --num_epochs 200 --model_type transformer --num_head 8 --d_k 64 --d_ff 2048 --src_pos_emb False --tgt_pos_emb True --max_relative_pos 64 --use_neg_dist True --nlayers 6 --trans_drop 0.2 --dropout_emb 0.2 --dropout 0.2 --early_stop 20 --warmup_epochs 0 --warmup_steps 2000 --optimizer adam --learning_rate 1e-4 --lr_decay 0.95 --valid_metric perfect --checkpoint True --use_bpe True --node_type_tag False --submodel_dim 512 --MTL False --constant_weight 1 0 --singleToken True --use_seq True --use_gnn True --copy_attn False --use_cuda True --template True --repeat_mode True --STL-learning False --temp_rate 1 --pretrain_stage False --from_pretrain None --dicts /root/CCSG/data/data_bpe/dicts_java_block.pkl --train_dataset /root/CCSG/data/data_bpe/train_java_block_entire.pkl --test_dataset /root/CCSG/data/data_bpe/eval_java_block_entire.pkl            var=1
            break
        fi
        count=$(($count+1))    
    done    
done