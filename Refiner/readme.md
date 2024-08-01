###Training Models:

train.py --data_workers [number os data workers]

-dataset_name [dataset name]

--data_dir [dataset path]

--model_dir [model path]

--model_name [model name]

--train_cache [train cache file path]

--train_tgt [train target file path]

--train_gnn [train code graph path]

--dev_cache [dev cache file path]

--dev_tgt [dev target file path]

--dev_gnn [dev code graph path]

--uncase [Whether to ignore case]

--use_src_word [Whether use word embedding]

--max_src_len [max length of source code]

--max_tgt_len [max length of target snippet]

--emsize [the size of embedding]

--src_vocab_size [the size of source code vocabulary]

--tgt_vocab_size [the size of target snippet vocabulary]

--share_decoder_embeddings [whether share the embeddings in decoder]

--batch_size [train batch size]

--test_batch_size [eval batch size]

--num_epochs [number of max epochs]

--num_head [number of heads in attention]

--d_k [dimension of key in attention]

--d_v [dimension of value in attention]

--d_ff [dimension of feed forward]

--src_pos_emb [whether use position embedding in source code]

--tgt_pos_emb [whether use position embedding in target]

--max_relative_pos [the max relative position]

--use_neg_dist [whether use negative Max value for relative position representations]

--nlayers [number of layers in transformer]

--trans_drop [drop out in transformer]

--dropout_emb [drop out in embedding]

--dropout [drop out in other cases]

--copy_attn [whether use copy mechanism]

--early_stop [early stop epochs]

--warmup_steps [the number of warm up steps]

--optimizer [type of optimizer]

--learning_rate 0.0001

--lr_decay 0.99 

--valid_metric bleu [the valid metric to save the best checkpoint]

--checkpoint [whether save]

--use_bpe [whether use BPE]

--MTL [whether use multi-task learning]

--constant_weight 0.9 0.1 [the coresponding weight of main task and auxilary task]

--node_type_tag [when embedding whether use the node type as tag]

--submodel_dim [the dimension of regreesor]  -

-use_seq [whether use code sequence encoder]

--use_gnn [whether use code graph encoder]

--singleToken [the single-token prediction mode]