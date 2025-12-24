# pip install -q transformers
import argparse
import os

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

arg_parser = argparse.ArgumentParser(description="")
arg_parser.add_argument('--model_name',type=str)
arg_parser.add_argument('--in_file',type=str)
arg_parser.add_argument('--out_file',type=str)
arg_parser.add_argument('--device',type=str)
arg_parser.add_argument('--batch_size',type=int)

args=arg_parser.parse_args()

checkpoint = args.model_name
device = args.device # for GPU usage or "cpu" for CPU usage


tokenizer = AutoTokenizer.from_pretrained(checkpoint,use_auth_token="",trust_remote_code=True,revision="",mirror='tuna')
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)

with open(args.in_file) as f:
    alllines=f.readlines()

tokenizer.pad_token = "<fim-middle>"

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
    pos_b=find_all_occurrences(text,"<|endoftext|>")
    if len(pos_b) == 0:
        pos_b=-1
    else:
        pos_b=pos_b[0]
    if len(find_all_occurrences(text,"<fim-middle>")) == 0:
        pos_a = pos_b
    else:
        pos_a=find_all_occurrences(text,"<fim-middle>")[-1] + len("<fim-middle>")
    return text[pos_a:pos_b]
if os.path.exists(args.out_file):
    with open(args.out_file, mode="r") as f:
        begin_num = len(f.readlines())
else:
    begin_num = 0

span=args.batch_size
with open(args.out_file,mode="a") as f:
    for begin_idx in tqdm(range(begin_num,len(alllines),span)):
        if begin_idx+span>len(alllines):
            tar_lines=alllines[begin_idx:len(alllines)]
        else:
            tar_lines=alllines[begin_idx:begin_idx+span]
        input_list = ["<fim-prefix>"+i.split("\t")[0].replace("<extra_id_0>","<fim-suffix>") +" <fim-middle>" for i in tar_lines]
        inputs = tokenizer.batch_encode_plus(input_list, return_tensors="pt",max_length=420,padding='max_length',truncation='longest_first').to(device)
        outputs = model.generate(inputs['input_ids'],max_new_tokens=200,attention_mask=inputs['attention_mask'])
        hyp = tokenizer.batch_decode(outputs)
        assert len(tar_lines)==len(hyp)
        for src_text,hyp_text in zip(tar_lines,hyp):
            src_text = src_text.split("\t")[0]
            hyp_text=hyp_text.replace("\t","").replace("\n","")
            final_hyp=get_target_span(hyp_text)
            f.write(src_text+"\t"+final_hyp+"\n")
