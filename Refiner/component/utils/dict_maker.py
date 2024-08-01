import logging
import random
import string
from collections import Counter
from tqdm import tqdm
import pickle
import os.path as osp
from typing import List
import re
import codecs
from typing import Iterator, Any
import gzip
import json
import subprocess
import sys
sys.path.append("..")
from component.inputters.utils import load_words_pain
from component.inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary
import torch
from tokenizers.implementations import BertWordPieceTokenizer as BPETokenizer
from component.inputters.constants import BOS_WORD, EOS_WORD, PAD_WORD, \
    UNK_WORD,BOS_S_WORD,STRING_TAG, TOKEN_TYPE_MAP, AST_TYPE_MAP, DATA_LANG_MAP, LANG_ID_MAP, NODE_TYPE_MAP

REF_BEGIN_WORD = "<ref>"
REF_END_WORD="</ref>"

class Args:
    def __init__(self,src_vocab_size):
        self.src_vocab_size=src_vocab_size


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])

def load_data(filename, data_dir,args):

    token_voc_path = osp.join(data_dir,  "tokenizer.pkl")
    type_voc_path = osp.join(data_dir, "type_node.dict")
    lines=[]
    for f_name in filename:
        with open(f_name,mode="r") as f:
            lines.extend(f.readlines())


    has_token_vocab = osp.exists(token_voc_path)
    has_type_vocab = osp.exists(type_voc_path)


    token_words = []
    type_words = []
    code_list=[]
    exs=[]

    for text in tqdm(lines):
        exs.append(text)
        line_graph = json.loads(text)

        token_seq = [e[1] for e in line_graph['edges']["InField"]]
        # token_seq = sorted(token_seq)
        pointer=0
        code_list.append(line_graph['text'])
        for key, node in enumerate(line_graph['nodes']):
            if pointer<len(token_seq) and int(key)==token_seq[pointer]:
                token_words.append(str(node))
                pointer+=1
            else:
                type_words.append(str(node))

    print(len(token_words))
    if not has_token_vocab:
        bpe_tk = BPETokenizer(unk_token=UNK_WORD, pad_token=PAD_WORD, lowercase=False)
        bpe_tk.add_tokens([PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD, REF_BEGIN_WORD,REF_END_WORD, BOS_S_WORD, STRING_TAG])
        bpe_tk.train_from_iterator(token_words, vocab_size=args.src_vocab_size,
                                   special_tokens=[PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD, REF_BEGIN_WORD,REF_END_WORD, BOS_S_WORD,
                                                   STRING_TAG])
        with open(token_voc_path, 'wb') as f:
            pickle.dump(bpe_tk, f)

    if not has_type_vocab:
        words_type = load_words_pain(args, type_words)
        dictionary_type = UnicodeCharsVocabulary(words_type,
                                                30,
                                                no_special_token=True)
        with open(type_voc_path, 'wb') as f:
            pickle.dump(dictionary_type, f)

    return exs

args=Args(src_vocab_size=80000)
load_data(["/root/LLM/full_data/train_java_construct_hgt_tokenlist.txt","/root/LLM/full_data/test_java_construct_hgt_tokenlist.txt"],"/root/LLM/vocab_dir/80000_new",args)