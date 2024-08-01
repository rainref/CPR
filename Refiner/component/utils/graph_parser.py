import tree_sitter
from tree_sitter import Language, Parser
from tqdm import tqdm
import json
JAVA_LANGUAGE = Language('/root/chatgpt/tree-sitter/tree-sitter/build/my-languages.so', 'java')

parser = Parser()
parser.set_language(JAVA_LANGUAGE)


id_dict={}
max_num=0
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




def trav_tree_make_number(root_node,begin_pos=-1,end_pos=-1,now_idx=-1,id_dict={},range_marker=[]):
    if not root_node.is_named:
        assert len(root_node.children)==0
        return now_idx,id_dict,range_marker
    if str(root_node) in id_dict:
        return now_idx, id_dict, range_marker
    now_idx+=1
    id_dict[str(root_node)]=now_idx
    if root_node.start_point[1] >= begin_pos and root_node.end_point[1] <= end_pos:
        range_marker.append(now_idx)
    if root_node.type != 'string_literal':
        for child in root_node.children:
            now_idx,_,_=trav_tree_make_number(child,now_idx=now_idx,id_dict=id_dict,begin_pos=begin_pos,end_pos=end_pos,range_marker=range_marker)
    return now_idx,id_dict,range_marker

def generate_node_text(root_node,id_dict,result_dict={},string_dict={},sub_mapping=None):
    res_dict={}
    result_dict[str(root_node)]=res_dict
    if sub_mapping is None:
        sub_mapping={}
    if len(root_node.children)>0:
        if root_node.type == 'string_literal':
            text = root_node.text.decode()
            if text not in string_dict:
                string_dict[text] = "STRING{}".format(len(string_dict))
            text = string_dict[text]
            res_dict['value']=[(text,root_node.start_point[1])]
            for child in root_node.children:
                sub_mapping[str(child)]=str(root_node)
            return result_dict
        maybe_children =[]
        for child in root_node.children:
            generate_node_text(child,id_dict,result_dict,string_dict,sub_mapping)
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
            parent_dict['value'].append((root_node.text.decode(),root_node.start_point[1]))
            return result_dict
        else:
            res_dict['value']=[(root_node.text.decode(),root_node.start_point[1])]
    res_dict['type']=root_node.type
    return  result_dict

def parse_text(text,begin_pos=-1,end_pos=-1,sp_res=None):
    tree = parser.parse(bytes(text.encode()))
    now_idx,id_dict,range_marker=trav_tree_make_number(tree.root_node,begin_pos=begin_pos,end_pos=end_pos,id_dict={},range_marker=[])
    string_dict,result_dict={},{}
    res=generate_node_text(tree.root_node,id_dict,result_dict=result_dict,string_dict=string_dict)
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
        if token_list[idx][1] <= begin_pos and token_list[idx+1][1] >= begin_pos:
            begin_idx=idx
        if token_list[idx][1] <= end_pos and token_list[idx+1][1] >= end_pos:
            end_idx = idx
    token_list.insert(end_idx,("</ref>",end_pos))
    token_list.insert(begin_idx,("<ref>", begin_pos))

    return res_by_order,range_marker,token_list,answer_text


const_len=len("public class test { ")
error_counter=0

with open("/root/chatgpt/train_java_token_data.tsv",mode="r",encoding="utf-8") as f:
    allline=f.readlines()
with open("train_java_token_graph_tokenlist.tsv",mode="w",encoding="utf-8") as f_out:
    for index,line in enumerate(tqdm(allline)):
        sp = line.split("	")
        if len(sp) < 2: continue
        elif len(sp)==3:
            sp_src,sp_res,sp_ref=sp
            sp_src,sp_res,sp_ref=json.loads(sp_src),json.loads(sp_res),json.loads(sp_ref)
            sp0 =  "public class test { " + sp_src + "}"
            new_text = sp0.replace("<extra_id_0>", sp_ref.replace("\n", ""))
            begin_pos=sp0.find("<extra_id_0>")
            end_pos=begin_pos+len(sp_ref.replace("\n", ""))
            res_text,range_marker,token_list,answer_text=parse_text(new_text,begin_pos,end_pos,sp_res)
            result = {
                "nodes": res_text,
                "line_id": index,
                "mark_range": range_marker,
                "text":" ".join([i[0] for i in token_list]),
                # 'text': sp0.replace("<extra_id_0>", " <ref> " + sp_ref.replace("\n", "") + " </ref> "),
                "answer": answer_text

            }

            f_out.write(str(result)+"\n")