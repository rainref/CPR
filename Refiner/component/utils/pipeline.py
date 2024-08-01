from tqdm import tqdm
import json
import tree_sitter
from tree_sitter import Language, Parser
from tqdm import tqdm
import json
JAVA_LANGUAGE = Language('/root/chatgpt/tree-sitter/tree-sitter/build/my-languages.so', 'java')

parser = Parser()
parser.set_language(JAVA_LANGUAGE)
import pickle

import argparse
arg_parser = argparse.ArgumentParser(description="")
arg_parser.add_argument('--ref_file',type=str)
arg_parser.add_argument('--answer_file',type=str)
arg_parser.add_argument('--out_file',type=str)
arg_parser.add_argument('--max_hyp_len',type=int)
arg_parser.add_argument('--no_ref',action='store_true')
args=arg_parser.parse_args()


id_dict={}
max_num=0

ref_file=args.ref_file
answer_file=args.answer_file
out_file=args.out_file

# convert format
if not args.no_ref:
    with open(answer_file,mode="r",encoding="utf-8") as f:
        allline_gen=f.readlines()
else:
    print("no ref")
    allline_gen=[]


with open(ref_file,mode="r",encoding="utf-8") as f:
    allline_ans=f.readlines()

def convert_format(allline_gen,allline_ans):
    gen_dict={}
    if args.no_ref:
        for line in tqdm(allline_ans):
            src,answer=line.split("\t")
            gen_dict[src]=(answer,"MASK")
    else:
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

    format_lines=[]
    for key,value in gen_dict.items():
        if not isinstance(value,tuple):
            continue
        text = json.dumps(key) + "\t" + json.dumps(value[0].replace("\n",""))+ "\t" + json.dumps(value[1].replace("\n",""))
        format_lines.append(text)
    return format_lines

def trav_tree_make_text(root_node,res_list=None,string_dict=None):
    global max_num
    if len(root_node.children)==0:
        res_list.append((root_node.text.decode(),root_node.start_point[1]))
    elif root_node.type == 'string_literal':
        node_text=root_node.text.decode()
        if node_text not in string_dict:
            string_dict[node_text] = "STRING{}".format(len(string_dict))
        res_list.append((string_dict[node_text] ,root_node.start_point[1]))
    else:
        for child in root_node.children:
            trav_tree_make_text(child,res_list,string_dict)




def trav_tree_make_number(root_node,begin_pos=-1,end_pos=-1,now_idx=-1,id_dict={},range_marker=[],attr_dict={}):
    attr_dict[str(root_node)] = False
    if root_node.start_point[1] >= begin_pos and root_node.end_point[1] <= end_pos:
        attr_dict[str(root_node)] = True
    if not root_node.is_named:
        assert len(root_node.children)==0
        return now_idx,id_dict,range_marker,attr_dict
    if str(root_node) in id_dict:
        return now_idx, id_dict, range_marker,attr_dict
    now_idx+=1
    id_dict[str(root_node)] = now_idx

    if root_node.type != 'string_literal':
        for child in root_node.children:
            now_idx,_,_,_=trav_tree_make_number(child,now_idx=now_idx,id_dict=id_dict,begin_pos=begin_pos,end_pos=end_pos,range_marker=range_marker,attr_dict=attr_dict)
    return now_idx,id_dict,range_marker,attr_dict

def generate_node_text(root_node,id_dict,result_dict={},string_dict={},sub_mapping=None,attr_dict={}):
    if str(root_node) in attr_dict and attr_dict[str(root_node)]:
        prefix="f!:"
    else:
        prefix=""
    res_dict={}
    result_dict[str(root_node)]=res_dict
    if sub_mapping is None:
        sub_mapping={}
    if len(root_node.children)>0:
        if root_node.type == 'string_literal':
            text = root_node.text.decode()
            if text not in string_dict:
                string_dict[text] = "STRING{}".format(len(string_dict))
            text = prefix+string_dict[text]
            res_dict['value']=[(text,root_node.start_point[1])]
            for child in root_node.children:
                sub_mapping[str(child)]=str(root_node)
            return result_dict
        maybe_children =[]
        for child in root_node.children:
            generate_node_text(child,id_dict,result_dict,string_dict,sub_mapping,attr_dict)
            if str(child) in id_dict:
                maybe_children.append(id_dict[str(child)])
        if len(maybe_children)>0:
            res_dict['children']=maybe_children
    else:
        if not root_node.is_named:
            parent_text=str(root_node.parent)
            if parent_text not in result_dict:
                parent_text=sub_mapping[parent_text]
            parent_dict=result_dict[parent_text]
            if "value" not in parent_dict:
                parent_dict['value']=[]
            parent_dict['value'].append((prefix+root_node.text.decode(),root_node.start_point[1]))
            return result_dict
        else:
            res_dict['value']=[(prefix+root_node.text.decode(),root_node.start_point[1])]
    res_dict['type']=prefix+root_node.type
    return  result_dict

def parse_text(text,begin_pos=-1,end_pos=-1,sp_res=None):
    tree = parser.parse(bytes(text.encode()))
    now_idx,id_dict,range_marker,attr_dict=trav_tree_make_number(tree.root_node,begin_pos=begin_pos,end_pos=end_pos,id_dict={},range_marker=[])
    string_dict,result_dict={},{}
    res=generate_node_text(tree.root_node,id_dict,result_dict=result_dict,string_dict=string_dict,attr_dict=attr_dict)
    answer_list=[]
    tree_answer = parser.parse(bytes(sp_res.encode()))
    trav_tree_make_text(tree_answer.root_node,answer_list,string_dict)
    answer_list.sort(key=lambda x: x[1])
    answer_text=" ".join([i[0] for i in answer_list])

    id_tuple=list(id_dict.items())
    id_tuple.sort(key=lambda x:x[1])
    id_dict_order=[i[0] for i in id_tuple]
    res_by_order=[res[i] for i in id_dict_order]
    token_list=[]
    for item in res_by_order:
        if "value" in item:
            token_list.extend(item['value'])
            # item['value']=[i[0] for i in item['value']]
    token_list.sort(key=lambda x : x[1])
    begin_idx=0
    end_idx=-1
    for idx in range(len(token_list)-1):
        if token_list[idx][1] <= begin_pos and token_list[idx+1][1] > begin_pos:
            begin_idx=idx
        if token_list[idx-1][1] < end_pos and token_list[idx][1] >= end_pos:
            end_idx = idx
    token_list.insert(end_idx,("</ref>",end_pos))
    token_list.insert(begin_idx,("<ref>", begin_pos))
    

    return res_by_order,range_marker,token_list,answer_text


def graph_parser(allline):
    const_len=len("public class test { ")
    error_counter=0
    res_line=[]
    for index,line in enumerate(tqdm(allline)):
        sp = line.split("	")
        if len(sp) < 2: continue
        elif len(sp)==3:
            sp_src,sp_res,sp_ref=sp
            sp_src,sp_res,sp_ref=json.loads(sp_src),json.loads(sp_res),json.loads(sp_ref)
            sp0 =  "public class test { " + sp_src + "}"
            if len(sp_ref)>args.max_hyp_len:
                sp_ref=sp_ref[:args.max_hyp_len]
            new_text = sp0.replace("<extra_id_0>", sp_ref.replace("\n", ""))
            begin_pos=sp0.find("<extra_id_0>")
            end_pos=begin_pos+len(sp_ref.replace("\n", ""))
            res_text,range_marker,token_list,answer_text=parse_text(new_text,begin_pos,end_pos,sp_res)
            result = {
                "nodes": res_text,
                "line_id": index,
                "mark_range": range_marker,
                "origin_text": sp0,
                "text":" ".join([i[0].replace("f!:","") for i in token_list]),
                # 'text': sp0.replace("<extra_id_0>", " <ref> " + sp_ref.replace("\n", "") + " </ref> "),
                "answer": answer_text

            }

            res_line.append(str(result))
    return res_line

def data_convertor(lines):
    reses_line=[]
    for line in tqdm(lines):
        json_line = eval(line)
        if isinstance(json_line, dict):
            old_json_line = json_line
            code_index = json_line['line_id']
            mark_range = json_line['mark_range']
            text_code = json_line['text']
            answer_data = json_line['answer']
            origin_text = json_line['origin_text']
            json_line = json_line["nodes"]
        else:
            code_index = None
        new_data = {"nodes": [], "edges": {
            "Child": [],
            "InField": [],
            "NextToken": [],
            "NextUse": []
        },
                    # "token_seq":[],
                    # "type_seq":[],
                    "token_type_map": []}
        token_node = []
        type_node = []
        last_token = -1
        old_new_mapping = {}
        last_use_mapping = {}
        idx = 0
        if isinstance(json_line, str):
            json_line = json.loads(json_line)
        for origin_idx, nd in enumerate(json_line):
            if nd == 0:
                break
            if "type" not in nd:
                type_str = "string_literal"
            else:
                type_str = nd['type']
            type_idx = idx
            old_new_mapping[origin_idx] = type_idx
            new_data["nodes"].append(type_str)
            # new_data["type_seq"].append(type_idx)
            idx = idx + 1

            if "children" in nd.keys():
                childrens = nd["children"]
                for c in childrens:
                    new_data["edges"]["Child"].append([type_idx, -c])

            if "value" in nd.keys():
                values = nd["value"]
                if not isinstance(values, list):
                    values = [values]
                for v in values:
                    token_idx = idx
                    new_data["nodes"].append(v)
                    idx = idx + 1
                    # new_data["token_seq"].append(token_idx)
                    new_data["edges"]["InField"].append([type_idx, token_idx])
                    if last_token != -1:
                        new_data["edges"]["NextToken"].append([last_token, token_idx])
                    last_token = token_idx
                    if v in last_use_mapping.keys():
                        new_data["edges"]["NextUse"].append([last_use_mapping[v], token_idx])
                        last_use_mapping[v] = token_idx
                    else:
                        last_use_mapping[v] = token_idx
                    new_data["token_type_map"].append(type_idx)

        for e in new_data["edges"]["Child"]:
            e[1] = old_new_mapping[-e[1]]

        if code_index is not None:
            new_data['index'] = code_index
        new_data['mark_range'] = mark_range
        new_data['text'] = text_code
        new_data['answer'] = answer_data
        new_data['origin_text'] = origin_text

        reses_line.append(json.dumps(new_data))
    return reses_line

def make(lines):
    new_results=[]
    for line in tqdm(lines):
        obj_line = eval(line)
        nodes_tuples = [(node, idx) for idx, node in enumerate(obj_line["nodes"])]
        tokens_idx = [i[1] for i in obj_line['edges']['InField']]
        tokens_idx.sort()
        token_nodes=[]
        type_nodes=[]
        pointer=0
        for i in range(len(nodes_tuples)):
            if pointer < len(tokens_idx) and i == tokens_idx[pointer]:
                token_nodes.append(nodes_tuples[i])
                pointer+=1
            else:
                type_nodes.append(nodes_tuples[i])
        token_nodes.sort(key=lambda x:x[0][1])
        idx_mapping={}
        counter=0
        for node in type_nodes+token_nodes:
            idx_mapping[node[1]]=counter
            counter+=1
        new_res={}
        new_res['origin_text']=obj_line['origin_text']
        new_res['tokens'] = [i[0][0] for i in token_nodes]
        new_res['types'] = [i[0] for i in type_nodes]
        new_res['mark_range'] = []
        new_res['edges']={}
        for key in obj_line['edges']:
            new_res['edges'][key]=[]
            for l in obj_line['edges'][key]:
                new_list=(idx_mapping[l[0]],idx_mapping[l[1]])
                new_res['edges'][key].append(new_list)
        new_res['text']=(obj_line['text'],obj_line['answer'])
        for idx_k in range(0,len(new_res['types'])+len(new_res['tokens'])):
            if idx_k < len(new_res['types']):
                if new_res["types"][idx_k][:3]=="f!:":
                    new_res["types"][idx_k]=new_res["types"][idx_k][3:]
                    new_res['mark_range'].append(idx_k)
            else:
                if new_res['tokens'][idx_k-len(new_res['types'])][:3]=="f!:":
                    new_res['tokens'][idx_k-len(new_res['types'])]=new_res['tokens'][idx_k-len(new_res['types'])][3:]
                    new_res['mark_range'].append(idx_k)

        new_results.append(new_res)
    return new_results

format_lines=convert_format(allline_gen,allline_ans)
parser_lines=graph_parser(format_lines)
converted_lines=data_convertor(parser_lines)
final_result=make(converted_lines)
pickle.dump(final_result, open(out_file, mode="wb"))