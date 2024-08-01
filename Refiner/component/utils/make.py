print("hi")
import pickle
n_counter=0
with open("/root/LLM/full_data/train_java_token_hgt_tokenlist.txt", mode="r",encoding="utf-8") as f:
    line = f.readline()
    new_results=[]
    while line:
        obj_line = eval(line)
        print(n_counter)
        n_counter += 1
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
        new_res['tokens'] = [i[0][0] for i in token_nodes]
        new_res['types'] = [i[0] for i in type_nodes]
        new_res['edges']={}
        for key in obj_line['edges']:
            new_res['edges'][key]=[]
            for l in obj_line['edges'][key]:
                new_list=(idx_mapping[l[0]],idx_mapping[l[1]])
                new_res['edges'][key].append(new_list)
        new_res['text']=(obj_line['text'],obj_line['answer'])
        new_res['mark_range']=sorted([idx_mapping[i] for i in obj_line['mark_range']])
        new_results.append(new_res)
        line = f.readline()
    pickle.dump(new_results,open("/root/LLM/full_data/train_java_token_oldmode_tokenlist.pkl",mode="wb"))