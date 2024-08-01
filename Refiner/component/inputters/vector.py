import torch
import torch.nn.functional as F
import numpy as np
from component.inputters.constants import edge_type
import time

#GNN point2
def vectorize(ex, model,debug=False):
    """Vectorize a single example."""

    tgt_dict = model.tgt_dict
    gnn_dict = model.gnn_dict

    targetCode, gnn,targetCode_origin = ex['targetCode'], ex['gnn'],ex['targetCode_origin']


    vectorized_ex = dict()

    # vectorized_ex['code_tokens'] = code.tokens
    # test_tokens = code.tokens
    vectorized_ex['code_tokens'] = gnn.code_tokens
    vectorized_ex['code_char_rep'] = None
    vectorized_ex['code_type_rep'] = None
    vectorized_ex['code_mask_rep'] = None
    vectorized_ex['use_code_mask'] = False
    vectorized_ex['code_text']=' '.join(gnn.code_tokens).replace("@@ ","")
    # vectorized_ex['code_word_rep'] = torch.LongTensor(code.vectorize(word_dict=src_dict))
    vectorized_ex['code_word_rep'] = torch.LongTensor(gnn.code_vectorize(word_dict=gnn_dict))


    vectorized_ex['tar'] = None
    vectorized_ex['tar_tokens'] = None
    vectorized_ex['stype'] = None
    vectorized_ex['tar_word_rep'] = None
    vectorized_ex['tar_char_rep'] = None
    vectorized_ex['target'] = None

    if targetCode is not None:
        vectorized_ex['tar'] = targetCode_origin.text if targetCode_origin is not None else targetCode.text
        vectorized_ex['tar_tokens'] = targetCode.tokens
        vectorized_ex['stype'] = targetCode.type
        vectorized_ex['tar_word_rep'] = torch.LongTensor(targetCode.vectorize(word_dict=tgt_dict))
        if model.args.use_tgt_char:
            vectorized_ex['tar_char_rep'] = torch.LongTensor(targetCode.vectorize(word_dict=tgt_dict, _type='char'))
        # target is only used to compute loss during training
        vectorized_ex['target'] = torch.LongTensor(targetCode.vectorize(tgt_dict))

    vectorized_ex['edges'] = None
    vectorized_ex['nodes'] = None
    vectorized_ex['backbone_sequence'] = None
    vectorized_ex['adjacency_matrix'] = None
    # time_start = time.time()
    if gnn is not None:
        vectorized_ex['edges'] = gnn.edges
        vectorized_ex['nodes'] = torch.LongTensor(gnn.vectorize(word_dict=gnn_dict))
        vectorized_ex['backbone_sequence'] = gnn.backbone_sequence
        if model.args.node_type_tag:
            vectorized_ex['type_sequence']=gnn.type_sequence
        else: vectorized_ex['type_sequence'] = None
        if model.args.MTL:
            vectorized_ex['mask_id']=gnn.mask_id
            vectorized_ex["virtual_index"]=gnn.virtual_index
            vectorized_ex['mask_to_index']=[]
            vectorized_ex['to_mask_index'] = []
            for edge in gnn.edges:
                if edge[0]=="NextToken":
                    if edge[1]==gnn.mask_id:
                        vectorized_ex['mask_to_index'].append(edge[2])
                    if edge[2]==gnn.mask_id:
                        vectorized_ex['to_mask_index'].append(edge[1])


        # vectorized_ex['adjacency_matrix'] = gnn.getmatrix(edge_type)
        vectorized_ex['adjacency_matrix'] = torch.from_numpy(gnn.getmatrix(edge_type))
    # time_end = time.time()
    # print('one example cost', time_end - time_start)

    vectorized_ex['src_vocab'] = gnn.src_vocab
    # test = code.src_vocab
    # vectorized_ex['src_vocab'] = code.src_vocab
    vectorized_ex['use_src_word'] = model.args.use_src_word
    vectorized_ex['use_tgt_word'] = model.args.use_tgt_word
    vectorized_ex['use_src_char'] = model.args.use_src_char
    vectorized_ex['use_tgt_char'] = model.args.use_tgt_char
    vectorized_ex['use_code_type'] = model.args.use_code_type

    if debug:
        vectorized_ex['gnn']=ex['gnn']
    else: vectorized_ex['gnn'] = None
    return vectorized_ex


def batchify_old(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    use_src_word = batch[0]['use_src_word']
    use_tgt_word = batch[0]['use_tgt_word']
    use_src_char = batch[0]['use_src_char']
    use_tgt_char = batch[0]['use_tgt_char']
    use_code_type = batch[0]['use_code_type']
    use_code_mask = batch[0]['use_code_mask']
    gnns=None
    if batch[0]['gnn'] is not None:
        gnns=[ex['gnn'] for ex in batch]

    # time_start = time.time()
    # --------- Prepare GNN tensors ---------
    batch_nodes = [ex['nodes'] for ex in batch]
    lengths_node = [len(node) for node in batch_nodes]
    nodes = torch.zeros(batch_size, max(lengths_node)).long()  # node feature
    length_pad = max(lengths_node)
    for i, node in enumerate(batch_nodes):
        length = lengths_node[i]
        nodes[i,:length] = node[:length]

    if batch[0]['type_sequence'] is not None:
        type_sequence = torch.zeros(batch_size, max(lengths_node)).long()
        for index,b in enumerate(batch):
            t_s=b['type_sequence']
            type_sequence[index,:len(t_s)]=torch.tensor(t_s)
    else: type_sequence=None

    if "mask_id" in batch[0].keys():
        mask_id_matrix=torch.tensor([b['mask_id'] for b in batch])
        virtual_index_matrix=torch.tensor([b['virtual_index'] for b in batch])
        mask_relative={"mask_to":[b['mask_to_index'] for b in batch],"to_mask":[b['to_mask_index'] for b in batch]}
    else:
        mask_id_matrix=torch.tensor([0 for b in batch])
        virtual_index_matrix = torch.tensor([0 for b in batch])
        mask_relative = {"mask_to": [0 for b in batch], "to_mask": [0 for b in batch]}

    adjacency_matrix = []

    for b in batch:
        matrix = b['adjacency_matrix']
        type_matrixs = []
        for i in matrix.split(matrix.shape[0], dim=-1):
            p2d = (0, length_pad - i.shape[0], 0, length_pad - i.shape[0])
            newmatrix = F.pad(i, p2d, "constant", 0) # i:[72,72],p2d:(0,118,0,118)左右上下填充
            type_matrixs.append(newmatrix)
        adjacency_matrix.append(torch.cat(type_matrixs, dim=-1))
    adjacency_matrix = torch.stack(adjacency_matrix, dim=0).float()
    batch_backbone_sequence = [ex['backbone_sequence'] for ex in batch]  # backbone_sequence

    lengths_backbone = [len(backbone) for backbone in batch_backbone_sequence]

    backbone_sequence = torch.zeros(batch_size, max(lengths_backbone), dtype=torch.long)  # backbone_sequence feature
    for i, backbone in enumerate(batch_backbone_sequence):
        length = lengths_backbone[i]
        backbone_sequence[i,:length] = torch.tensor(backbone[:length])
    lengths_backbone = torch.tensor(lengths_backbone)  # backbone length
    lengths_node = torch.tensor(lengths_node)  # backbone length
    # time_end = time.time()
    # print('one batch cost', time_end - time_start)

    # --------- Prepare Code tensors ---------
    code_words = [ex['code_word_rep'] for ex in batch]

    max_code_len = max([d.size(0) for d in code_words])


    # Batch Code Representations
    code_len_rep = torch.zeros(batch_size, dtype=torch.long)
    code_word_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_src_word else None


    source_maps = []
    src_vocabs = []
    for i in range(batch_size):
        # code_len_rep[i] = code_words[i].size(0)
        code_len_rep[i] = code_words[i].size(0)
        if use_src_word:
            code_word_rep[i, :code_words[i].size(0)].copy_(code_words[i])


        context = batch[i]['code_tokens']
        vocab = batch[i]['src_vocab']
        src_vocabs.append(vocab)
        # Mapping source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([vocab[w] for w in context])
        source_maps.append(src_map)

    # --------- Prepare targetCode tensors ---------
    no_targetCode = batch[0]['tar_word_rep'] is None
    if no_targetCode:
        tar_len_rep = None
        tar_word_rep = None
        tar_char_rep = None
        tgt_tensor = None
        alignments = None
    else:
        tar_words = [ex['tar_word_rep'] for ex in batch]
        tar_chars = [ex['tar_char_rep'] for ex in batch]
        max_sum_len = max([q.size(0) for q in tar_words])
        if use_tgt_char:
            max_char_in_tar_token = tar_chars[0].size(1)

        tar_len_rep = torch.zeros(batch_size, dtype=torch.long)
        tar_word_rep = torch.zeros(batch_size, max_sum_len, dtype=torch.long) \
            if use_tgt_word else None
        tar_char_rep = torch.zeros(batch_size, max_sum_len, max_char_in_tar_token, dtype=torch.long) \
            if use_tgt_char else None

        max_tgt_length = max([ex['target'].size(0) for ex in batch])
        tgt_tensor = torch.zeros(batch_size, max_tgt_length, dtype=torch.long)
        alignments = []
        for i in range(batch_size):
            tar_len_rep[i] = tar_words[i].size(0)
            if use_tgt_word:
                tar_word_rep[i, :tar_words[i].size(0)].copy_(tar_words[i])
            if use_tgt_char:
                tar_char_rep[i, :tar_chars[i].size(0), :].copy_(tar_chars[i])
            #
            tgt_len = batch[i]['target'].size(0)
            tgt_tensor[i, :tgt_len].copy_(batch[i]['target'])
            target = batch[i]['tar_tokens']
            align_mask = torch.LongTensor([src_vocabs[i][w] for w in target])
            alignments.append(align_mask)

    return {
        'batch_size': batch_size,
        'code_word_rep': code_word_rep,
        'code_len': code_len_rep,
        'tar_word_rep': tar_word_rep,
        'tar_char_rep': tar_char_rep,
        'tar_len': tar_len_rep,
        'tgt_seq': tgt_tensor,
        'code_tokens': [ex['code_tokens'] for ex in batch],
        'code_text': [ex['code_text'] for ex in batch],
        #'tar_text': [ex['tar'] for ex in batch],
        'tar_text': [ex['tar'] for ex in batch],
        'tar_tokens': [ex['tar_tokens'] for ex in batch],
        'src_vocab': src_vocabs,
        'src_map': source_maps,
        'alignment': alignments,
        'stype': [ex['stype'] for ex in batch],

        'nodes_feature': nodes,
        'adjacency_matrix': adjacency_matrix,
        'backbone_sequence': backbone_sequence,
        'lengths_backbone': lengths_backbone,
        'lengths_node':lengths_node,
        'gnns':gnns,
        'type_sequence':type_sequence,
        'mask_id_matrix':mask_id_matrix,
        'virtual_index_matrix':virtual_index_matrix,
        'mask_relative':mask_relative,
        'eval_types':[ex['eval_type'] for ex in batch]
    }

def batchify(batch):

    lengths_src = torch.tensor([len(b['text_src']) for b in batch],dtype=torch.long)
    lengths_tgt = torch.tensor([len(b['text_tgt']) for b in batch],dtype=torch.long)
    batch_size = torch.tensor(len(batch),dtype=torch.long)


    vec_src = torch.zeros(batch_size, max(lengths_src)).long()
    vec_tgt = torch.zeros(batch_size, max(lengths_tgt)).long()

    raw_text=[]
    raw_tgt=[]

    alignments = None
    src_map = None


    for i, b in enumerate(batch):
        vec_src[i, :lengths_src[i]] = b['text_src']
        vec_tgt[i, :lengths_tgt[i]] = b['text_tgt']
        raw_text.append(b['raw_text'])
        raw_tgt.append(b['raw_tgt'])




    vec_type, vec_token, vec_attrs, edge_metrix, lengths_type, lengths_token, lengths_node,vec_mark_range=batchify_only_graph(batch)

    data = (vec_type,vec_token,vec_src,vec_tgt,vec_attrs,None,None,vec_mark_range),edge_metrix,(
    lengths_type,lengths_token, lengths_src , lengths_tgt,lengths_node,None
    ),(None,None,None)
    return {
        "batch_size":batch_size,
        "data":data,
        "raw_text":raw_text,
        "raw_tgt":raw_tgt,
        "raw_template":None
    }


def batchify_only_graph(batch):
    lengths_type = torch.tensor([len(b['types']) for b in batch],dtype=torch.long)
    lengths_token = torch.tensor([b['tokens'].shape[0] for b in batch],dtype=torch.long)
    lengths_node=torch.tensor([len(b['types'])+len(b['tokens']) for b in batch],dtype=torch.long)
    batch_size = torch.tensor(len(batch),dtype=torch.long)
    dims_token= torch.tensor([b['tokens'].shape[1] for b in batch],dtype=torch.long)

    vec_type=torch.zeros(batch_size, max(lengths_type)).long()
    vec_token = torch.zeros(batch_size, max(lengths_token),max(dims_token)).long()
    vec_mark_range=torch.zeros(batch_size, max(lengths_node)).long()

    adjacency_dict = {}
    for i, b in enumerate(batch):
        vec_type[i,:lengths_type[i]]=b['types']
        dim = b['tokens'].shape[1]
        vec_token[i, :lengths_token[i],:dim] = b['tokens']
        vec_mark_range[i,:lengths_node[i]]=b['mark_range']

        max_len=max(lengths_node)
        for edge_key in edge_type.keys():
            if edge_key not in b['edge_dicts'].keys():
                edge_metrix = torch.zeros([1, 1])
            else:
                edge_metrix = b['edge_dicts'][edge_key]
            if edge_key not in adjacency_dict.keys():
                adjacency_dict[edge_key] = []
                adjacency_dict["{}_T".format(edge_key)] = []

            edge_metrix=edge_metrix+torch.eye(edge_metrix.shape[0],edge_metrix.shape[0])
            edge_metrix_T=edge_metrix.T

            p2d = (0, max_len - edge_metrix.shape[0], 0, max_len - edge_metrix.shape[1])
            newmatrix = F.pad(edge_metrix, p2d, "constant", 0)
            newmatrix_T = F.pad(edge_metrix_T, p2d, "constant", 0)
            adjacency_dict[edge_key].append(newmatrix)
            adjacency_dict["{}_T".format(edge_key)].append(newmatrix_T)

    for key,v in adjacency_dict.items():
        mt=torch.stack(v,dim=0)
        adjacency_dict[key]=mt

    # for key in edge_key

    edge_ins=[]
    edge_outs = []
    for key in edge_type.keys():
        e=adjacency_dict[key]
        e_T=adjacency_dict["{}_T".format(key)]
        edge_ins.append(e)
        edge_outs.append(e_T)

    edge_ins=torch.stack(edge_ins,dim=-1)
    edge_outs = torch.stack(edge_outs, dim=-1)
    edge_metrix = torch.cat((edge_ins,edge_outs),dim=-1)
    edge_metrix = edge_metrix.reshape([batch_size,max(lengths_node),-1])

    return vec_type,vec_token,None,edge_metrix,lengths_type,lengths_token, lengths_node,vec_mark_range
