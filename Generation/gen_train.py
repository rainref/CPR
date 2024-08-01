from tqdm import tqdm
import re
import openai
import pickle
import json
import os

def ask_gpt(messages):
    if len(messages)>2:
        return "LIMIT"
    try:
        openai.api_key = ""
        res=openai.ChatCompletion.create(
          model="gpt-4o",
          messages=messages,
            timeout=30)
        return res['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return ask_gpt(messages)
        
code_dict={}
with open("train_android_construct.tsv",mode="r") as f:
    line=f.readline()
    while line:
        sp=line.split("	")
        sp1=sp[1].replace("\n","")
        pos=sp[0].find(" <extra_id_0>")
        origin_text=sp[0].replace(" <extra_id_0>",sp1)
        text=origin_text.replace(" ","")
        if text not in code_dict:
            code_dict[text]=[]
        code_dict[text].append((sp[0],sp1,pos,origin_text))
        code_dict[text].sort(key=lambda x:x[2],reverse=True)
        line=f.readline()
        
new_code_dict={}
for key,v in code_dict.items():
    new_code_dict[v[0][3]]=v
    
    
code_dict=new_code_dict


prompt='''Fill in {} missing parts of the following code, which are occupied by <1> to <{}> respectively.  You just need to output all the missing pieces, without repetition or explanation. The result needs to be formated in json, with keys from 1 to {} and the corresponding completion as their value. For example, {}.
###
{}'''
prompt_only_one='''Fill in the missing span at <1> in the following java code snippet. You only need to indicate the missing part, but not to explain or repeat the code.
###
{}'''



def generate_example(number):
    example_word=['code','res','example','find',"completion","demo","cache","field","target","span","prediction"]
    res_dict={}
    for i in range(number):
        res_dict[i+1]=example_word[i%11]
    return json.dumps(res_dict)
    
def make_prompt(origin_text,line_list):
    idx=len(line_list)
    for line_data in line_list:
        begin_pos=line_data[2]
        end_pos=line_data[2]+len(line_data[1])
        origin_text=origin_text[:begin_pos]+f"<{idx}>"+origin_text[end_pos:]
        idx-=1
    if len(line_list)==1:
        prompt_code=prompt_only_one.format(origin_text)

    else:
        prompt_code=prompt.format(len(line_list),len(line_list),len(line_list),generate_example(len(line_list)),origin_text)
    return prompt_code
    
    
def try_parse_result(res_text,list_len):
    if list_len == 1:
        return {1:res_text}
    try:
        res_json=json.loads(res_text)
    except Exception as e:
        return "An error occurred when the result was parsed into json:{}. Please re-output. You still don't need to explain, but simply return the result that can be parsed as json.".format(e)
    res_dict={}
    if not isinstance(res_json,dict):
        return "The result is not a dictionary in json format, please re-output. You still don't need to explain, but simply return the result that can be parsed as json."
    for k,v in res_json.items():
        ks=re.findall("\d+", str(k))
        if len(ks)==0 or len(ks) > list_len or len(ks) < 1:
            return "The key: {} in json, does not match any identifier of missing parts. Please re-output the complete json result, with keys range between 1-{}.".format(k,list_len)
        res_dict[ks[0]]=v
    for idx in range(1,list_len+1):
        if str(idx) not in res_dict:
            return "Missing part <{}> in the code snippet has no corresponding key in JSON. Re-output the complete result, making sure that the identifier of missing parts 1-{} have their corresponding keys in the result.".format(idx,list_len)
    res=dict(sorted(res_dict.items(),key=lambda x:x[0]))
    return res
    
target_file="train_android_construct.pkl"
if not os.path.exists(target_file):
    result_pkl={}
else:
    result_pkl=pickle.load(open(target_file,mode="rb"))

reses_line=[]
mark_len=len(" <extra_id_0>")
for line_id,(origin,line_list) in enumerate(tqdm(code_dict.items())):
    if origin in result_pkl:
        continue
    prompt_code=make_prompt(origin,line_list)
#     if len(re.findall("<\d*<\d*>\d*>", prompt_code)) > 0:
# #         print(prompt_code)
#         print(re.findall("<\d*<\d*>\d*>", prompt_code))
    msg=[
                {"role": "user", "content": prompt_code},
            ]
    res=ask_gpt(msg)
    try_res=try_parse_result(res,len(line_list))
    while isinstance(try_res,str):
        msg.append({"role": "assistant", "content": res})
        msg.append({"role": "user", "content": try_res})
        res=ask_gpt(msg)
        if res=="LIMIT":
            try_res=dict(zip([i for i in range(len(line_list))],["unk" for i in range(len(line_list))]))
            print("limit condition")
            break
        try_res=try_parse_result(res,len(line_list))
    result_pkl[origin]=try_res
    if line_id % 100==0:
        pickle.dump(result_pkl,open(target_file,mode="wb"))
    
    
pickle.dump(result_pkl,open(target_file,mode="wb"))