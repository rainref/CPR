import tqdm
import json
src_file=open("train_java_token_graph_tokenlist.tsv",mode="r",encoding="utf-8")
out_file=open("/root/LLM/full_data/train_java_token_hgt_tokenlist.txt",mode="w",encoding="utf-8")



if __name__ == '__main__':
    line=src_file.readline()
    counter=0
    while line:
        json_line=eval(line)
        if isinstance(json_line,dict):
            old_json_line=json_line
            code_index=json_line['line_id']
            mark_range=json_line['mark_range']
            text_code = json_line['text']
            answer_data=json_line['answer']
            json_line = json_line["nodes"]
        else:
            code_index=None
        new_data={"nodes":[],"edges":{
            "Child":[],
            "InField":[],
            "NextToken":[],
            "NextUse":[]
        },
                  # "token_seq":[],
                  # "type_seq":[],
                  "token_type_map":[]}
        token_node=[]
        type_node=[]
        last_token=-1
        old_new_mapping={}
        last_use_mapping={}
        idx=0
        if isinstance(json_line,str):
            json_line=json.loads(json_line)
        for origin_idx,nd in enumerate(json_line):
            if nd==0:
                break
            if "type" not in nd:
                type_str="string_literal"
            else:type_str=nd['type']
            type_idx=idx
            old_new_mapping[origin_idx]=type_idx
            new_data["nodes"].append(type_str)
            # new_data["type_seq"].append(type_idx)
            idx=idx+1

            if "children" in nd.keys():
                childrens=nd["children"]
                for c in childrens:
                    new_data["edges"]["Child"].append([type_idx,-c])

            if "value" in nd.keys():
                values = nd["value"]
                if not isinstance(values,list):
                    values=[values]
                for v in values:
                    token_idx = idx
                    new_data["nodes"].append(v)
                    idx = idx + 1
                    # new_data["token_seq"].append(token_idx)
                    new_data["edges"]["InField"].append([type_idx, token_idx])
                    if last_token!=-1:
                        new_data["edges"]["NextToken"].append([last_token,token_idx])
                    last_token=token_idx
                    if v in last_use_mapping.keys():
                        new_data["edges"]["NextUse"].append([last_use_mapping[v], token_idx])
                        last_use_mapping[v] = token_idx
                    else:last_use_mapping[v]=token_idx
                    new_data["token_type_map"].append(type_idx)

        for e in new_data["edges"]["Child"]:
            e[1]=old_new_mapping[-e[1]]

        if code_index is not None:
            new_data['index']=code_index
        new_data['mark_range']=mark_range
        new_data['text']=text_code
        new_data['answer']=answer_data

        out_file.write(json.dumps(new_data)+"\n")
        print(counter)
        counter=counter+1
        line = src_file.readline()

    print("end")
    out_file.close()
    src_file.close()






