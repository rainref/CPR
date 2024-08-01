from tree_sitter import Language, Parser
from tqdm import tqdm
import openai
import json
import os

import argparse
arg_parser = argparse.ArgumentParser(description="")
arg_parser.add_argument('--in_file',type=str)
arg_parser.add_argument('--out_file',type=str)
args=arg_parser.parse_args()


JAVA_LANGUAGE = Language('tree-sitter/tree-sitter/build/my-languages.so', 'java')

parser = Parser()
parser.set_language(JAVA_LANGUAGE)

origin_prompt = '''Fill in the missing span at <1> in the following java code snippet. You only need to indicate the missing part, but not to explain or repeat the code. 
{}
'''
new_prompt = """Puting your prediction back in context can get the code:
{code_text}
There is a syntax error at positions {begin_pos}-{end_pos}: [{text}] 
Please re-output the missing code in the <1> without repeating the context or giving any explanation."""


def trav_node_find_error(node):
    if not node.has_error:
        return None
    error_node = None
    for child in node.children:
        error_node = trav_node_find_error(child)
        if error_node is not None:
            return error_node
    if error_node is None:
        return node


const_len = len("public class test { ")


def ask_gpt(messages):
    try:
        openai.api_base = "https://api.openai-proxy.com/v1"
        openai.api_key = ""
        res = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages
        )
        return res['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return ask_gpt(messages)


error_counter = 0
with open(args.in_file, mode="r", encoding="utf-8") as f:
    allline = f.readlines()

output_file = args.out_file

if os.path.exists(output_file):
    with open(output_file, mode="r", encoding="utf-8") as f_out:
        now_lines = f_out.readlines()
else:
    now_lines = []
with open(output_file, mode="a", encoding="utf-8") as f_out:
    print("begin from {}".format(len(now_lines)))
    for line in tqdm(allline[len(now_lines):]):
        sp = line.split("	")
        if len(sp) < 2: continue
        new_text = sp[0].replace("<extra_id_0>", sp[1].replace("\n", ""))
        parsed_tree = "public class test { " + new_text + "}"
        tree = parser.parse(bytes(parsed_tree, "utf-8"))
        error_node = trav_node_find_error(tree.root_node)
        if error_node is not None:
            if len(new_text) > 1000:
                new_text = new_text[:1000]
            if len(sp[1]) > 500:
                sp1 = sp[1][:500]
            else:
                sp1 = sp[1]
            if len(error_node.text) > 500:
                error_text = error_node.text[:500]
            else:
                error_text = error_node.text

            specific_prompt = new_prompt.format(code_text=new_text, begin_pos=error_node.start_point[1] - const_len,
                                                end_pos=error_node.end_point[1] - const_len, text=error_text.decode("utf8","ignore"))
            msgs = [
                {"role": "user", "content": origin_prompt.format(sp[0].replace("<extra_id_0>", "<1>"))},
                {"role": "assistant", "content": sp1.replace("\n", "")},
                {"role": "user", "content": specific_prompt}
            ]
            res = ask_gpt(msgs)
        else:
            res = sp[1]
        res = res.replace("\n", "").replace("\t", "")
        f_out.write(sp[0] + "\t" + res + "\n")
