import openai
from tqdm import tqdm
import time
def ask_gpt_data(messages):
    if len(messages)>4:
        return "LIMIT"
    try:
        openai.api_key = ""
        res=openai.ChatCompletion.create(
          model="gpt-4o",
          messages=messages
        )
        return res['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return ask_gpt_data(messages)
    
def ask_gpt(text):
    return ask_gpt_data([{"role": "user", "content": text}])
prompt='''Fill in the missing span at <1> in the following java code snippet. You only need to indicate the missing part, but not to explain or repeat the code. 
{}
'''
input_file="test_java_token.tsv"
output_file="test_java_token_gpt.tsv"
with open(input_file,mode="r") as f:
    all_lines=f.readlines()
with open(output_file,mode="r",encoding="utf-8") as f_out:
    now_lines=f_out.readlines()
with open(output_file,mode="a",encoding="utf-8") as f_out:
    print("begin from {}".format(len(now_lines))) 
    for line_data in tqdm(all_lines[len(now_lines):]):
        split_data=line_data.split("	")
        query_text=split_data[0].replace("<extra_id_0>","<1>")
        prompt_text=prompt.format(query_text)
        res=ask_gpt(prompt_text)
        res=res.replace("\n","")
        f_out.write(split_data[0]+"	"+res+"\n")
    
