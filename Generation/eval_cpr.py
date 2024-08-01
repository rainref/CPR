import pickle
from component.eval import metrics
from tree_sitter import Language, Parser
from tqdm import tqdm
import openai
import os
import json
import subprocess
JAVA_LANGUAGE = Language('/root/chatgpt/tree-sitter/tree-sitter/build/my-languages.so', 'java')

parser = Parser()
parser.set_language(JAVA_LANGUAGE)
# import argparse
# arg_parser = argparse.ArgumentParser(description="")
# arg_parser.add_argument('--ref_path',type=str)
# arg_parser.add_argument('--answer_path',type=str)
# args=arg_parser.parse_args();

import argparse
arg_parser = argparse.ArgumentParser(description="")
arg_parser.add_argument('--ref_path',dest="ref_path",nargs='+',help="list")
args=arg_parser.parse_args();

ref_paths=args.ref_path
tokenizer=pickle.load(open("/root/LLM/full_data/dicts.pkl.new",mode="rb"))
vocab_token=tokenizer['dictionary_token']
# ref_path="/root/S2RCC/data/test_java_construct.tsv"
# answer_path="/root/chatgpt/test_java_construct_chatgpt_regen.tsv"

for ref_path in ref_paths:
    with open(ref_path,mode="r") as f:
        refs=f.readlines()

    rp_tuples=[("+ +","++"),("! =","!="),("= =","=="),("> =",">="),("< =","<="),("- >","->"),("| |","||"),("& &","&&"),("$ $","$$"),("/ =","/="),("< <","<<"),("> >",">>"),(" . ","."),("- - >","-->"),(" - -","--"),("& ","&")]
    def repair_text(text):
        for tp in rp_tuples:
            tp0,tp1=tp
            text=text.replace(tp0,tp1)
        return text
    candidate_replace=[(" _ ","_"),(" _ "," _"),(" _ ","_ ")]
    def try_parse(text):
        text = repair_text(text)
        for tp in candidate_replace:
            tp0,tp1=tp
            tree=parser.parse(bytes(text.replace(tp0,tp1),"utf-8"))
            if trav_node_find_error(tree.root_node) is None:
                return True
        return False

    def to_bpe(text):
        nn = vocab_token.pre_tokenizer.pre_tokenize_str(text)
        raw_tgt=[i[0].lower() for i in nn]
        return " ".join(raw_tgt)
    def trav_node_find_error(node):
        if not node.has_error:
            return None
        error_node=None
        for child in node.children:
            error_node = trav_node_find_error(child)
            if error_node is not None:
                return error_node
        if error_node is None:
            return node

    import sys
    sys.path.append(".")
    sys.path.append("./main")
    from main.train import eval_official_scores
    ref_dict={}
    pred_dict={}
    error_counter=0
    all_counter=0
    final_dict=[]
    for al in tqdm(refs):
        all_counter+=1
        al=eval(al)
        code_text=al['code'].replace("<extra_id_0>",al['hyp'])
        ref_text=al['code'].replace("<extra_id_0>",al['ref'])
        final_dict.append((ref_text,code_text))
        if not try_parse(code_text):
            error_counter+=1

        ref_dict[al['code']]=to_bpe(al['ref'])
        pred_dict[al['code']]=to_bpe(al['hyp'])

    bleu1,bleu2,bleu3,bleu4,lv,perfect=eval_official_scores(ref_dict,pred_dict)
    with open(ref_path+".eval.txt",mode="w") as f:
        print("hyp_file",ref_path,file=f)
        print("error_rate:",error_counter/all_counter,file=f)
        print("bleu1:",bleu1,file=f)
        print("bleu2:",bleu2,file=f)
        print("bleu3:",bleu3,file=f)
        print("bleu4:",bleu4,file=f)
        print("lv:",lv,file=f)
        print("perfect:",perfect,file=f)

        with open("/root/LLM/full_data/评估文件/refs.txt",mode="w") as f_ref:
            with open("/root/LLM/full_data/评估文件/ans.txt",mode="w") as f_ans:
                for ref,ans in tqdm(final_dict):
                    f_ref.write(ref+"\n")
                    f_ans.write(ans+"\n")

        sys.path.append("/root/S2R-LLM-O/LLM/codebleu")
        command = r'python3 /root/S2R-LLM-O/LLM/codebleu/calc_code_bleu.py --refs /root/LLM/full_data/评估文件/refs.txt --hyp /root/LLM/full_data/评估文件/ans.txt --lang java'
        output = subprocess.check_output(command,shell=True)
        print(output,file=f)
     
    
    