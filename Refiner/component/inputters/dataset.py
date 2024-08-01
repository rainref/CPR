# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/data.py

import random

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from component.inputters.vector import vectorize
import json
from component.inputters import utils
from component.inputters.utils import get_node_representation
from .constants import edge_type
from .constants import EOS_WORD,BOS_WORD,EOS,BOS
import torch
from component.inputters.vocabulary import Vocabulary
from pretreat.pretreatment import replace_type,not_replace_type,deep_in_tree,getAttr,build_edges
import random
from javalang import tokenizer
from javalang.parser import Parser
import re

BUFSIZE = 409600000  # 400MB
def from_ex_to_item(ex,args,is_eval,vocab_token,vocab_sep,vocab_type,vocab_attr,vocab_identi=None,is_micro=False):


    src_map = None
    alignments = None

    node_tokens, node_types, node_attr, edge_dicts,mark_range = get_node_representation(ex, vocab_token,
                                                                                 vocab_type, vocab_attr,
                                                                                 pretrain=args.pretrain_stage)
    src_text=ex["text"][0]
    if args.no_ref:
        tar_text = re.findall(r"<ref>.*</ref>",ex['text'][0])[0]
        src_text = src_text.replace(tar_text,"<ref>")

    text_src = torch.tensor(vocab_token.encode(src_text).ids,
                            dtype=torch.int64)
    text_tgt = BOS_WORD + " " + ex['text'][1] +" "+ EOS_WORD


    text_tgt = torch.tensor(vocab_token.encode(text_tgt).ids,
                            dtype=torch.int64)

    if len(text_tgt) > args.max_tgt_len:
        print("cut off tgt{}".format(len(text_tgt)))
        text_tgt=text_tgt[:args.max_tgt_len]


    text_of_raw=ex['text'][1]

    nn = vocab_token.pre_tokenizer.pre_tokenize_str(text_of_raw)
    raw_tgt=[i[0] for i in nn]





    item = {
        "tokens": node_tokens,
        "types": node_types,
        "edge_dicts": edge_dicts,
        "MASK_id": None,
        "text_src": text_src,
        "text_tgt": text_tgt,
        "attrs": node_attr,
        "src_vocab": None,
        "src_map": src_map,
        "alignments": alignments,
        "raw_text": ex['origin_text'],
        "raw_tgt": raw_tgt,
        "raw_template": None,
        "sep_tgt": None,
        "mark_range":mark_range,
    }
    return item


class DataLoader(object):
    def __init__(self, filename, batch_size, max_len):
        self.batch_size = batch_size
        self.max_len = max_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

    def __iter__(self):
        lines = self.stream.readlines(BUFSIZE)
        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)

        # for json
        docs = []
        for line in lines:
            line = eval(line)
            line = line['content']
            line = [self.tokenizer.tokenize(l.strip()) for l in line.split('\n') if l.strip()]
            docs.append(line)
        docs = [x for x in docs if len(x) > 2]  # 筛掉一些短文章，很关键
        random.shuffle(docs)
        # end for json

        data = []
        for idx, doc in enumerate(docs):
            data.extend(self.create_instances_from_document(docs, idx, self.max_len))

        idx = 0
        while idx < len(data):
            yield self.convert_to_features(data[idx:idx + self.batch_size], self.tokenizer, self.encode_type,
                                           self.max_len)
            idx += self.batch_size
# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------
class InCompleteCodeDataset(Dataset):
    def __init__(self, examples,args,vocabs,is_eval=False,is_micro=False):
        self.examples = examples
        self.args=args
        (self.vocab_attr, self.vocab_type, self.vocab_token,self.vocab_sep,self.vocab_identi)=vocabs
        self.is_eval=is_eval
        self.is_micro=is_micro

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item=self.convert_vectors(self.examples[index])
        return item

    def lengths(self):
        return len(self.examples)

    def convert_vectors(self,ex):
        item=from_ex_to_item(ex, self.args, self.is_eval, self.vocab_token, self.vocab_sep, self.vocab_type, self.vocab_attr,self.is_micro)
        if len(item['text_src'])>self.args.max_src_len:
            print("cut off{}".format(len(item['text_src'])))
            item['text_src']=item['text_src'][:self.args.max_src_len]
        return item



class PretrainDataset(Dataset):
    def __init__(self, examples,args,vocabs,is_eval=False):
        self.examples = examples
        self.args=args
        (self.vocab_attr, self.vocab_type, self.vocab_token,self.vocab_sep,self.vocab_identi)=vocabs
        self.is_eval = False

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item=self.convert_vectors(self.examples[index])


        return item

    def lengths(self):
        return len(self.examples)

    def build_edges(self,res_list):
        res = {
            "tokens": [],
            "types": []
        }

        id_mapping = {}
        now_id = 0

        for i in res_list["types"]:
            id_mapping[getAttr(i, "id")] = now_id
            now_id = now_id + 1
        for i in res_list["tokens"]:
            id_mapping[getAttr(i, "id")] = now_id
            now_id = now_id + 1


        for i in res_list["tokens"]:
            attr = i["belong_attr"]
            res["tokens"].append((i["value"], attr))

        return res

    def convert_vectors(self,ex):
        ex=ex.copy()
        origin_text=ex['text']
        method_begin=0
        for idx,i in enumerate(origin_text):
            if i.position.column==ex["begin_pos"]:
                method_begin=idx
                break
        max_len=len(ex['text'])-method_begin
        # max_len = max_len if max_len < 30 else 30
        # random_number = random.randint(5, max_len) if max_len > 5 else random.randint(0, max_len)
        all_len_15 = int(max_len * 0.15)
        random_number=all_len_15
        begin_point=random.randint(method_begin, len(ex['text']) - random_number)

        end_point=begin_point+random_number

        full_text=[]
        parse_text=[]
        predict_tar=[]
        first_appear = False
        for idx,i in enumerate(origin_text):
            if begin_point <= idx < end_point:
                predict_tar.append(i)
                if not first_appear:
                    full_text.append(tokenizer.Identifier("MASK", position=i.position))
                    first_appear = True
                if type(i) in replace_type or i.value in replace_type:
                    parse_text.append(tokenizer.Identifier("MASK", position=i.position))
                elif type(i) in not_replace_type:
                    parse_text.append(i)

            else:
                full_text.append(i)
                parse_text.append(i)
        ex['text']=([i.value for i in full_text],predict_tar)

        parser = Parser(parse_text)
        tree = parser.parse()

        res_list = {
            "tokens": [],
            "types": []
        }
        deep_in_tree(tree.children[2][0], res_list, None)
        res=self.build_edges(res_list)
        if len(res['tokens']) != len(ex['tokens']):
            res = build_edges(res_list)
            ex['tokens'] = res['tokens']
            ex['types'] = res['types']
            ex['edges'] = res['edges']
        else:
            ex['tokens'] = res['tokens']

        item = from_ex_to_item(ex, self.args, self.is_eval, self.vocab_token, self.vocab_sep, self.vocab_type,
                               self.vocab_attr)
        if len(item['text_src'])>self.args.max_src_len:
            padding_len=int((self.args.max_src_len-1)/2)
            begin_pos=begin_point-padding_len
            if begin_pos<0:
                padding_len = padding_len-(begin_pos)
                begin_pos=0

            end_pos=begin_point+1+padding_len
            if end_pos>len(item['text_src']):
                end_pos=len(item['text_src'])
            item['text_src']=item['text_src'][begin_pos:end_pos]
            print("cut off{} to {}".format(begin_pos,end_pos))
        return item





class CombineDataset(Dataset):
    def __init__(self, examples, model,args,eval_types=None):
        self.model = model
        self.examples = examples[0]
        self.args=args
        self.eval_types=eval_types

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        json_object=json.loads(self.examples[index])
        args=self.args
        example=utils.process_examples(0,
                                   json_object['target'].replace('@@ ',""),
                                   json_object,
                                   args.max_src_len,
                                   args.max_tgt_len,
                                   args.code_tag_type,
                                   uncase=args.uncase,
                                   test_split=False,
                                   tgt_bpe=json_object['target'] if args.use_bpe else None, MTL=args.MTL)
        vectorized_ex = vectorize(example, self.model)
        vectorized_ex['eval_type']=self.eval_types[index] if self.eval_types is not None else None
        return vectorized_ex

    def lengths(self):
        return len(self.examples)

class CommentDataset(Dataset):
    def __init__(self, examples, model):
        self.model = model
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.model)

    def lengths(self):
        return [(len(ex['gnn'].code_tokens), len(ex['targetCode'].tokens))
                for ex in self.examples]


# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
