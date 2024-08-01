from typing import List
import sentencepiece
import torch
import tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import json
from tqdm import tqdm
import time

import argparse
arg_parser = argparse.ArgumentParser(description="")
arg_parser.add_argument('--model_name',type=str)
arg_parser.add_argument('--in_file',type=str)
arg_parser.add_argument('--out_file',type=str)
arg_parser.add_argument('--count_time',action='store_true')
arg_parser.add_argument('--device',type=str)
args=arg_parser.parse_args()
tokenizers_version = tuple(int(n) for n in tokenizers.__version__.split('.'))
if tokenizers_version < (0, 12, 1):
    print("warning: Your tokenizers version looks old and you will likely have formatting issues. We recommend installing tokenizers >= 0.12.1")

model_name=args.model_name
print("loading model")
model = AutoModelForCausalLM.from_pretrained(model_name)
print("loading tokenizer")
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
print("loading complete")
device=args.device

model = model.to(device)
model = model.eval()
with open(args.in_file) as f:
    alllines=f.readlines()
if args.count_time:
    alllines=alllines[:1000]
    args.out_file=args.out_file+".debug"
def find_all_occurrences(string, sub_string):
    positions = []
    start = 0
    while True:
        position = string.find(sub_string, start)
        if position == -1:
            break
        positions.append(position)
        start = position + 1
    return positions
def get_target_span(text):
    list_a=find_all_occurrences(text,"<|mask:0|>")
    if len(list_a)>0:
        pos_a=list_a[-1] + len("<|mask:0|>")
    else:
        pos_a=-1
    pos_b=[i for i in find_all_occurrences(text,"<|endofmask|>") if i>pos_a]
    if len(pos_b)>0:
        pos_b=pos_b[0]
    else:
        pos_b=-1

    return text[pos_a:pos_b]

span=25
tokenizer.padding_side = "left"
tokenizer.pad_token =  "<|endofmask|>"
all_time=0
with open(args.out_file,mode="w") as f:
    for begin_idx in tqdm(range(0,len(alllines),span)):
        if begin_idx+span>len(alllines):
            tar_lines=alllines[begin_idx:len(alllines)]
        else:
            tar_lines=alllines[begin_idx:begin_idx+span]
        input_list = [i.split("\t")[0].replace("<extra_id_0>","<|mask:0|>")+"<|/ file |><|mask:1|><|mask:0|>" for i in tar_lines]
        inputs = tokenizer.batch_encode_plus(input_list, return_tensors="pt",max_length=420,padding='max_length',truncation='longest_first').to(device)
        with torch.no_grad():
            time_begin=time.time()
            outputs = model.generate(inputs['input_ids'],max_new_tokens=100,attention_mask=inputs['attention_mask'])
            time_end=time.time()
            all_time+=time_end-time_begin
        hyp = tokenizer.batch_decode(outputs)
        assert len(tar_lines)==len(hyp)
        for src_text,hyp_text in zip(tar_lines,hyp):
            src_text = src_text.split("\t")[0]
            hyp_text=hyp_text.replace("\t","").replace("\n","")
            final_hyp=get_target_span(hyp_text)
            final_hyp=final_hyp.replace("\t","").replace("\n","")
            
            f.write(src_text+"\t"+final_hyp+"\n")
print("average time:{}".format(all_time/len(alllines)))