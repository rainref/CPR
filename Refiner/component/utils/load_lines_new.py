from tqdm import tqdm
import pickle
import torch
import sys
sys.path.append("../..")
from component.inputters.constants import edge_type


def get_text_representation(ex,vocab_token,vocab_type):
    text_ids = vocab_token.encode(ex['text']).ids
    answer_ids=vocab_token.encode("<s> "+ex['answer']+ " </s>").ids
    nn=vocab_token.pre_tokenizer.pre_tokenize_str(ex['answer'])
    raw_tgt = [i[0] for i in nn]
    return {
        "text":ex['text'],
        "answer":raw_tgt,
        "text_ids":text_ids,
        "answer_ids":answer_ids,
        "mark_range":ex['mark_range'],
        "id":ex['index']
    }

def get_node_representation(ex,vocab_token,vocab_type):
    # node_len = len(ex["tokens"]) + len(ex["types"])
    node_len = len(ex['nodes'])
    edge_dicts = {}
    for key in edge_type.keys():
        if key in ex['edges'].keys():
            es = ex['edges'][key]
        else:
            es = []
        edge_metrix = torch.zeros([node_len, node_len])
        if len(es) > 0:
            index_edges = torch.tensor([e for e in es], dtype=torch.long).T
            edge_metrix = edge_metrix.index_put((index_edges[0], index_edges[1]), torch.ones(index_edges.shape[1]))
        edge_dicts[key] = edge_metrix

    token_index=list(set([i[1] for i in ex['edges']['InField']]))
    token_index.sort()
    now_tk_idx=0
    tokens_node=[]
    types_node=[]
    for index,node in enumerate(ex['nodes']):
        if now_tk_idx<len(token_index) and  index==token_index[now_tk_idx]:
            tokens_node.append(node)
            now_tk_idx+=1
        else:
            types_node.append(node)

    tk_emb=[vocab_token.encode(i).ids for i in tokens_node]

    len_of_sub_token=[len(i) for i in tk_emb]
    max_len_of_sub=max(len_of_sub_token)

    max_number=3

    if max_len_of_sub > max_number:
        max_len_of_sub = max_number
    for idx,tk_list in enumerate(tk_emb):
        if len(tk_list) < max_len_of_sub:
            tk_list.extend([0 for i in range(max_len_of_sub-len(tk_list))])
        elif len(tk_list) > max_len_of_sub:
            tk_emb[idx] = tk_list[:max_len_of_sub]


    node_tokens = torch.tensor(tk_emb,
                               dtype=torch.int64)
    node_types = torch.tensor([vocab_type[i] for i in types_node],
                              dtype=torch.int64)
    node_hyp=torch.zeros(size=[len(ex['nodes'])],dtype=torch.int64)
    for mark_pos in ex['mark_range']:
        node_hyp[mark_pos]=1

    if len(token_index) != len(node_tokens):
        print("wait")


    return {"node_tokens":node_tokens,
            "node_types":node_types,
            "edge_dicts":edge_dicts,
            "node_hyp":node_hyp,
            "token_index":token_index
            }

def get_node_representation_hgt(ex,vocab_token,vocab_type):
    # node_len = len(ex["tokens"]) + len(ex["types"])
    node_type_sum=len(edge_type)
    from_idx, to_idx, edge_type_list = [], [], []
    for key,idx in edge_type.items():
        if key in ex['edges'].keys():
            es = ex['edges'][key]
        else:
            es = []
        for i in es:
            from_idx.extend([i[0],i[1]])
            to_idx.extend([i[1],i[0]])
            edge_type_list.extend([idx,idx+node_type_sum])

    edge_index=torch.stack([torch.tensor(from_idx,dtype=torch.int64),torch.tensor(to_idx,dtype=torch.int64)],dim=0)
    edge_type_list=torch.tensor(edge_type_list,dtype=torch.int64)


    token_index=list(set([i[1] for i in ex['edges']['InField']]))
    token_index.sort()
    now_tk_idx=0
    tokens_node=[]
    types_node=[]
    for index,node in enumerate(ex['nodes']):
        if now_tk_idx<len(token_index) and  index==token_index[now_tk_idx]:
            tokens_node.append(node)
            now_tk_idx+=1
        else:
            types_node.append(node)

    tk_emb=[vocab_token.encode(i).ids for i in tokens_node]

    len_of_sub_token=[len(i) for i in tk_emb]
    max_len_of_sub=max(len_of_sub_token)

    max_number=3

    if max_len_of_sub > max_number:
        max_len_of_sub = max_number
    for idx,tk_list in enumerate(tk_emb):
        if len(tk_list) < max_len_of_sub:
            tk_list.extend([0 for i in range(max_len_of_sub-len(tk_list))])
        elif len(tk_list) > max_len_of_sub:
            tk_emb[idx] = tk_list[:max_len_of_sub]


    node_tokens = torch.tensor(tk_emb,
                               dtype=torch.int64)
    node_types = torch.tensor([vocab_type[i] for i in types_node],
                              dtype=torch.int64)
    node_hyp=torch.zeros(size=[len(ex['nodes'])],dtype=torch.int64)
    for mark_pos in ex['mark_range']:
        node_hyp[mark_pos]=1

    if len(token_index) != len(node_tokens):
        print("wait")


    return {"node_tokens":node_tokens,
            "node_types":node_types,
            "edge_index":edge_index,
            "edge_type_list":edge_type_list,
            "node_hyp":node_hyp,
            "token_index":token_index
            }

def convert_data(file_name,vocab_token, vocab_type,length=0):
    result_data=[]
    pbar=tqdm()
    counter=0
    with open(file_name,mode="r") as f:
        line=f.readline()
        while line and counter <= length:
            if length!=0:
                counter+=1
            json_data=eval(line)
            representation={}
            node_rep=get_node_representation(json_data, vocab_token, vocab_type)
            text_rep=get_text_representation(json_data, vocab_token, vocab_type)
            representation.update(node_rep)
            representation.update(text_rep)
            result_data.append(representation)
            line = f.readline()
            pbar.update(1)
    return  result_data


if __name__ == "__main__":
    tokenizer=pickle.load(open("/root/LLM/vocab_dir/80000_new/tokenizer.pkl",mode="rb"))
    vocab_dict=pickle.load(open("/root/LLM/vocab_dir/80000_new/type_node.dict",mode="rb"))
    result_data=convert_data("/root/LLM/full_data/train_java_construct_hgt_tokenlist.txt",tokenizer,vocab_dict)
    pickle.dump(result_data,open("/root/LLM/full_data/train_java_construct_tokenlist.txt.cache",mode="wb"))