with open("/root/chatgpt/test_java_token_gpt_regen_result.tsv",mode="r",encoding="utf-8") as f:
    allline_gen=f.readlines()
with open("/root/chatgpt/test_java_token.tsv",mode="r",encoding="utf-8") as f:
    allline_ans=f.readlines()
from tqdm import tqdm
import json
gen_dict={}
for line in tqdm(allline_gen):
    sps=line.split("\t")
    if len(sps)!=2:
        continue
    src,gen_data=sps
    gen_dict[src]=gen_data
for line in tqdm(allline_ans):
    src,answer=line.split("\t")
    if src in gen_dict:
        gen_data = gen_dict[src]
        if isinstance(gen_data,tuple):
            continue
        gen_dict[src]=(answer,gen_data)

with open("/root/chatgpt/test_java_token_data.tsv",mode="w",encoding="utf-8") as f_out:
    for key,value in gen_dict.items():
        if not isinstance(value,tuple):
            continue
        text = json.dumps(key) + "\t" + json.dumps(value[0].replace("\n",""))+ "\t" + json.dumps(value[1].replace("\n",""))
        f_out.write(text+"\n")
print("wait")