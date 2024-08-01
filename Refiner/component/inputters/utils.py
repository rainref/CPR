import logging
import random
import string
from collections import Counter
from tqdm import tqdm
import pickle
import os.path as osp
from typing import List
import re
from component.inputters import constants as constants

from component.objects import Code, TargetCode, Graph
from component.inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary
from component.inputters.constants import BOS_WORD, EOS_WORD, PAD_WORD, \
    UNK_WORD,MASK_WORD,BOS_S_WORD,STRING_TAG, TOKEN_TYPE_MAP, AST_TYPE_MAP, DATA_LANG_MAP, LANG_ID_MAP, NODE_TYPE_MAP,REF_BEGIN_WORD,REF_END_WORD
from component.utils.misc import count_file_lines, count_gz_file_lines
from component.objects.BPE_util import BPEUtil
import codecs
from typing import Iterator, Any
import gzip
import json
import os
from pretreat.pretreatment import replace_type,not_replace_type
import torch
from .constants import edge_type
# import matplotlib.pyplot as plt
from tokenizers.implementations import BertWordPieceTokenizer as BPETokenizer


logger = logging.getLogger(__name__)


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def generate_random_string(N=8):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------
# node_num = []
def process_examples(lang_id,
                     target,
                     graph,
                     max_src_len,
                     max_tgt_len,
                     code_tag_type,
                     uncase=False,
                     test_split=True, tgt_bpe=None, MTL=False, virtual_index=None):
    code_type = []

    code_type = code_type[:max_src_len]

    TAG_TYPE_MAP = TOKEN_TYPE_MAP if \
        code_tag_type == 'subtoken' else AST_TYPE_MAP

    if tgt_bpe is not None:
        tarCode_bpe = tgt_bpe.lower() if uncase else tgt_bpe
        tarCode_tokens = tarCode_bpe.split()
        if not test_split:
            tarCode_tokens = tarCode_tokens[:max_tgt_len]
        if len(tarCode_tokens) == 0:
            return None
        sumarry_bpe = TargetCode()
        sumarry_bpe.text = ' '.join(tarCode_tokens)
        sumarry_bpe.tokens = tarCode_tokens
        sumarry_bpe.prepend_token(BOS_WORD)
        sumarry_bpe.append_token(EOS_WORD)
    else:
        sumarry_bpe = None
    if target is not None:
        tarCode = target.lower() if uncase else target
        tarCode_tokens = tarCode.split()
        if not test_split:
            tarCode_tokens = tarCode_tokens[:max_tgt_len]
        if len(tarCode_tokens) == 0:
            return None
        targetCode = TargetCode()
        targetCode.text = ' '.join(tarCode_tokens)
        targetCode.tokens = tarCode_tokens
        targetCode.prepend_token(BOS_WORD)
        targetCode.append_token(EOS_WORD)
    else:
        targetCode = None

    if graph is not None:
        gnn = Graph()

        nodes = []
        edges = []
        for key, value in graph['node_labels'].items():
            nodes.append(value)
        for edge_type, edge_list in graph['edges'].items():
            for edge in edge_list:
                edges.append([edge_type, edge[0], edge[1]])

        backbone_sequence = graph['backbone_sequence'][:max_src_len]
        # nodes_lower = ' '.join(nodes).lower().split()
        nodes_lower = [node.lower() for node in nodes]

        # global node_num
        # node_num.append(len(nodes_lower))
        gnn.edges = edges
        gnn.tokens = nodes_lower
        gnn.backbone_sequence = backbone_sequence
        if MTL:
            gnn.code_tokens.insert(0, PAD_WORD)

        if "type_sequence" in graph.keys():
            gnn.type_sequence = graph['type_sequence']
        else:
            gnn.type_sequence = None

        if "mask_id" in graph.keys():
            gnn.virtual_index = graph['virtual_index']
            gnn.mask_id = graph['mask_id']
        else:
            gnn.virtual_index = None
            gnn.mask_id=None

    else:
        gnn = None

    example = dict()

    example['targetCode_origin'] = targetCode if sumarry_bpe is not None else None
    example['gnn'] = gnn
    example['targetCode'] = sumarry_bpe if sumarry_bpe is not None else targetCode
    return example


def addNextTokenEdge(num_edge):
    edges = []
    for i in range(num_edge - 1):
        edge = ['NextToken', i, i + 1]
        edges.append(edge)
    return edges


def addNewConnection(graph, target_node, new_node):
    graph['node_labels'].append(new_node)
    new_id = len(graph['node_labels']) - 1

    for index, node in enumerate(graph['node_labels']):
        if node == target_node:
            graph['edges'].append(['Child', index, new_id])


def deleteNodes(graph, node_List, replace_node="<D>", layout=None):
    node_instance = []
    max_len = len(graph['node_labels'])
    node_ids = []
    for index, node in enumerate(graph['node_labels']):
        if node in node_List:
            node_ids.append(index)
        else:
            node_instance.append(node)
    node_ids.sort()

    replace_id = 0
    if replace_node in node_instance:
        replace_id = node_instance.index(replace_node)
    else:
        if replace_node == "<D>":
            replace_id = -1
        else:
            node_instance.append(replace_node)
            replace_id = len(node_instance) - 1

    Hash_after = [i for i in range(len(node_instance))]

    for it in node_ids:
        Hash_after.insert(it, replace_id)

    newEdges = []
    for index, edge in enumerate(graph['edges']):
        for i in range(1, 3):
            edge[i] = Hash_after[edge[i]]
        if edge[1] != edge[2] and edge[1] != -1 and edge[2] != -1:
            newEdges.append(edge)

    newBack = []
    for index, back_node in enumerate(graph['backbone_sequence']):
        back_node = Hash_after[back_node]
        if back_node != -1:
            if len(newBack) > 0:
                if newBack[-1] != back_node:
                    newBack.append(back_node)
            else:
                newBack.append(back_node)

    graph['node_labels'] = node_instance
    graph['edges'] = newEdges
    graph['backbone_sequence'] = newBack


def to_BPE(tar_list, graphs, args):
    bpeTool = BPEUtil(args.bpe_vocab)
    target_bpe = []
    for line in tar_list:
        bpe_line = bpeTool.bpe.process_line(line)
        target_bpe.append(bpe_line)
    for graph in graphs:
        maxoftoken = max(graph['backbone_sequence'])
        graph['edges']['Subtoken'] = []
        tokenList = [[id, token] for id, token in graph['node_labels'].items() if int(id) <= maxoftoken]
        type_mode = "type_sequence" in graph.keys()
        for [id, token] in tokenList:
            id = int(id)
            bpe_string = bpeTool.bpe.segment(token)
            if (bpe_string != token and token != "mask_code" and token != "MASK_CODE"):
                subTokens = bpe_string.split(" ")
                now_id = len(graph['node_labels'])
                last_node_id = [edge[0] for edge in graph['edges']['NextToken'] if edge[1] == id]
                next_node_id = [edge[1] for edge in graph['edges']['NextToken'] if edge[0] == id]
                if id not in graph['backbone_sequence']:
                    print("error")
                back_position = graph['backbone_sequence'].index(id)
                graph['backbone_sequence'].remove(id)
                for new_node in subTokens:
                    graph['node_labels'][str(now_id)] = new_node
                    if type_mode:
                        graph['type_sequence'].append(NODE_TYPE_MAP['Subtoken'])
                    graph['backbone_sequence'].insert(back_position, now_id)
                    graph['edges']['Subtoken'].append([id, now_id])
                    for n in last_node_id:
                        graph['edges']['NextToken'].append([n, now_id])
                    last_node_id = [now_id]
                    now_id = now_id + 1
                    back_position = back_position + 1
                for n in next_node_id:
                    graph['edges']['NextToken'].append([n, last_node_id[0]])

    return target_bpe


def add_node_type_sequence(graphs):
    for graph in graphs:
        graph['type_sequence'] = []
        for id, token in graph['node_labels'].items():
            if int(id) in graph['backbone_sequence']:
                graph['type_sequence'].append(NODE_TYPE_MAP['Token'])
            else:
                graph['type_sequence'].append(NODE_TYPE_MAP['AST'])


def add_virtual_nodes(graphs, max_virtual=20):
    for graph in graphs:
        less_id = len(graph['node_labels'])
        last_virtual = -1
        for t in range(less_id, less_id + max_virtual):
            graph['node_labels'][str(t)] = "mask_code"
            if last_virtual != -1:
                graph['edges']['NextToken'].append([last_virtual, t])
            last_virtual = t
            if "type_sequence" in graph.keys():
                graph['type_sequence'].append(NODE_TYPE_MAP['Subtoken'])
        max_id = len(graph['node_labels'])
        graph['virtual_index'] = [less_id, max_id]


def load_data_text(args, filenames):
    # create tag
    if filenames['gnn'] is not None:
        with open(filenames['gnn'], encoding='utf-8') as f:
            graphs = [line.strip() for line in
                      tqdm(f, total=count_file_lines(filenames['gnn']), desc='read gnn')]
            if args.debug:
                graphs = graphs[:args.debug_data_len]
            graphs = [json.loads(line) for line in tqdm(graphs, desc='convert line to json')]

    if filenames['tgt'] is not None:
        with open(filenames['tgt'], encoding='utf-8') as f:
            targets = [line.strip() for line in
                       tqdm(f, total=count_file_lines(filenames['tgt']))]
    else:
        targets = [None] * len(graphs)

    return {'graphs': graphs, 'targets': targets}


def load_data(args, filenames, max_examples=-1, dataset_name='java',
              test_split=False):
    """Load examples from preprocessed file. One example per line, JSON encoded."""

    # create tag
    if filenames['gnn'] is not None:
        with open(filenames['gnn'], encoding='utf-8') as f:
            graphs = [line.strip() for line in
                      tqdm(f, total=count_file_lines(filenames['gnn']), desc='read gnn')]
            if args.debug:
                graphs = graphs[:args.debug_data_len]
            graphs = [json.loads(line) for line in tqdm(graphs, desc='convert line to json')]

    if filenames['tgt'] is not None:
        with open(filenames['tgt'], encoding='utf-8') as f:
            targets = [line.strip() for line in
                       tqdm(f, total=count_file_lines(filenames['tgt']))]
    else:
        targets = [None] * len(graphs)

    if args.debug:
        targets = targets[:args.debug_data_len]
        graphs = graphs[:args.debug_data_len]

    if args.node_type_tag:
        add_node_type_sequence(graphs)


    # GNN point 4
    # graph=graphs[0]
    # deleteNodes(graph, ["mask", "code","MASK","CODE","MASK_CODE"], "mask_code")

    if args.MTL:
        for index, graph in enumerate(graphs):
            for key, value in graph['node_labels'].items():
                if value == "mask_code" or value == "MASK_CODE":
                    graph['mask_id'] = int(key)
                    continue
            if 'mask_id' not in graph.keys():
                print(index)
                graph['mask_id'] = graph['backbone_sequence'][-1]

    if args.use_bpe:
        targets_bpe = to_BPE(targets, graphs, args)
    else:
        targets_bpe = [None for i in range(len(targets))]

    if args.virtual:
        if args.singleToken:
            add_virtual_nodes(graphs, 5)
        else:
            add_virtual_nodes(graphs)

    examples = []
    for tgt, graph, tgt_bpe in tqdm(zip(targets, graphs, targets_bpe),
                                    total=len(graphs)):
        if dataset_name == 'java_bpe' or 'java_bpe_simple': dataset_name = "java"
        if dataset_name in ['java', 'python']:
            _ex = process_examples(LANG_ID_MAP[DATA_LANG_MAP[dataset_name]],
                                   tgt,
                                   graph,
                                   args.max_src_len,
                                   args.max_tgt_len,
                                   args.code_tag_type,
                                   uncase=args.uncase,
                                   test_split=test_split,
                                   tgt_bpe=tgt_bpe if args.use_bpe else None, MTL=args.MTL)
            if _ex is not None:
                examples.append(_ex)

        if max_examples != -1 and len(examples) > max_examples:
            break
    # global node_num
    # print(sum(node_num))
    # plt.ylim(0,2000)
    # plt.bar(range(0, len(node_num)), node_num)
    # plt.savefig(str(sum(node_num)) + '.jpg')
    return examples


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------


def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in tqdm(f, total=count_file_lines(embedding_file)):
            w = Vocabulary.normalize(line.rstrip().split(' ')[0])
            words.add(w)

    words.update([BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD])
    return words


def load_words(args, examples, fields, dict_size=None):
    """Iterate and index all the words in examples (documents + questions)."""

    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            words.append(w)
        word_count.update(words)

    word_count = Counter()
    for ex in tqdm(examples):
        json_obj=json.loads(ex)
        if fields==["gnn"]:
            words=[t for t in json_obj['node_labels'].values()]
            _insert(words)
        elif fields==["targetCode"]:
            words = [t for t in json_obj['target'].split(" ")]
            _insert(words)


    # -2 to reserve spots for PAD and UNK token
    dict_size = dict_size - 2 if dict_size and dict_size > 2 else dict_size
    most_common = word_count.most_common(dict_size)
    words = set(word for word, _ in most_common)
    return words


def load_words_pain(args, word_area, dict_size=None):
    """Iterate and index all the words in examples (documents + questions)."""

    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            words.append(w)
        word_count.update(words)

    word_count = Counter()
    _insert(word_area)

    # -2 to reserve spots for PAD and UNK token
    dict_size = dict_size - 2 if dict_size and dict_size > 2 else dict_size
    most_common = word_count.most_common(dict_size)
    words = set(word for word, _ in most_common)
    return words


def build_word_dict(args, examples, fields, dict_size=None,
                    no_special_token=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Vocabulary(no_special_token)
    for w in load_words(args, examples, fields, dict_size):
        word_dict.add(w)
    return word_dict


def build_word_and_char_dict(args, examples, fields, dict_size=None,
                             no_special_token=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    words = load_words(args, examples, fields, dict_size)
    dictioanry = UnicodeCharsVocabulary(words,
                                        args.max_characters_per_token,
                                        no_special_token)
    return dictioanry


def top_targetCode_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['targetCode'].tokens:
            w = Vocabulary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)


# create tag
def iteratate_jsonl_gz(filename: str) -> Iterator[Any]:
    reader = codecs.getreader('utf-8')
    with gzip.open(filename) as f:
        for line in reader(f):
            yield json.loads(line)


def init_to_cache_file(args, filenames, step=16):
    tgt_dict_path = osp.join(args.data_dir, args.dataset_name[0], "tgt.dict")
    graph_voc_path = osp.join(args.data_dir, args.dataset_name[0], "graph_node.dict")
    tgt_words = []
    graph_words = []
    results = []
    f_gnn = open(filenames['gnn'], encoding='utf-8')
    f_tgt = open(filenames['tgt'], encoding='utf-8')
    f_cache = open(filenames['cache'], encoding='utf-8', mode='w')
    total_line = count_file_lines(filenames['gnn'])
    if args.debug:
        total_line = 100
    pbar = tqdm(total=total_line, desc="process cache file")
    cache_gnn = []
    cache_tgt = []
    for i in range(total_line):
        pbar.update(1)
        line_gnn = json.loads(f_gnn.readline())
        line_tgt = convert_to_subtoken(f_tgt.readline().strip())
        cache_gnn.append(line_gnn)
        cache_tgt.append(line_tgt)
        if (i + 1) % step == 0 or (i + 1) == total_line:
            graphs = process_graphs(cache_gnn, cache_tgt, args)
            for graph in graphs:
                graph_words.extend([node for node in graph['node_labels'].values()])
                tgt_words.extend([subtoken for subtoken in graph['target'].split(" ")])
                json.dump(graph, f_cache)
                results.append(json.dumps(graph))
                f_cache.write("\n")
            cache_gnn = []
            cache_tgt = []
    if args.uncase:
        tgt_words = [word.lower() for word in tgt_words]
        graph_words = [word.lower() for word in graph_words]
    pbar.close()
    f_cache.close()

    if not osp.exists(tgt_dict_path):
        logger.info("creating new tgt dicts")
        words_tgt = load_words_pain(args, tgt_words, args.tgt_vocab_size)


        dictionary_tgt = UnicodeCharsVocabulary(words_tgt,
                                                args.max_characters_per_token,
                                                no_special_token=False)


        with open(tgt_dict_path, 'wb') as f:
            pickle.dump(dictionary_tgt, f)


    if not osp.exists(graph_voc_path):
        words_gnn = load_words_pain(args, graph_words, None)
        dictionary_gnn = UnicodeCharsVocabulary(words_gnn,
                                                args.max_characters_per_token,
                                                no_special_token=True)
        with open(graph_voc_path, 'wb') as f:
            pickle.dump(dictionary_gnn, f)

    return results

def subtokenizer(identifier: str) -> List[str]:
    # Tokenizes code identifiers
    splitter_regex = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    identifiers = re.split('[._\-]', identifier)
    subtoken_list = []

    for identifier in identifiers:
        matches = splitter_regex.finditer(identifier)
        for subtoken in [m.group(0) for m in matches]:
            subtoken_list.append(subtoken)

    return subtoken_list


def convert_to_subtoken(line):
    line = line.replace('==', ' == ')
    line = line.replace('!=', ' != ')
    line = line.replace('>=', ' >= ')
    line = line.replace('<=', ' <= ')
    line = line.replace(',', ' , ')
    line = line.replace('<', ' < ')
    line = line.replace('>', ' > ')
    line = line.replace('(', ' ( ')
    line = line.replace(')', ' ) ')
    line = line.replace('||', ' || ')
    line = line.replace('&&', ' && ')
    tok_list = subtokenizer(line)
    line = ' '.join(tok_list)
    return line

def load_cache_file(filenames, args):
    exs = []
    types = []
    f_cache = open(filenames['cache'], encoding='utf-8')
    if "type" in filenames.keys() and filenames['type'] is not None:
        f_type = open(filenames['type'], encoding='utf-8')
    else:
        f_type = None
    total = count_file_lines(filenames['cache'])
    for i in tqdm(range(total if not args.debug else 100)):
        cache_line = f_cache.readline()
        if f_type is not None:
            type_line = int(f_type.readline().replace("\n", ""))
        else:
            type_line = None
        if len(cache_line) < 50000 and len(cache_line) > 0 and '"target": ""' not in cache_line:
            exs.append(cache_line)
            types.append(type_line)
    logger.info('length of cache dataset = %d' % len(exs))
    return exs, types


def process_graphs(graphs, targets, args):
    if not isinstance(graphs, list):
        graphs = [graphs]
        targets = [targets]

    if args.node_type_tag:
        add_node_type_sequence(graphs)

    if args.MTL:
        for index, graph in enumerate(graphs):
            for key, value in graph['node_labels'].items():
                if value == "mask_code" or value == "MASK_CODE":
                    graph['mask_id'] = int(key)
                    continue
            if 'mask_id' not in graph.keys():
                print(index)
                graph['mask_id'] = graph['backbone_sequence'][-1]

    if args.use_bpe:
        targets_bpe = to_BPE(targets, graphs, args)
    else:
        targets_bpe = [None for i in range(len(targets))]

    if args.virtual:
        if args.singleToken:
            add_virtual_nodes(graphs, 5)
        else:
            add_virtual_nodes(graphs)
    else:
        add_virtual_nodes(graphs, 0)

    for index, g in enumerate(graphs):
        if args.use_bpe:
            g['target'] = targets_bpe[index]
        else:g['target'] = targets[index]

    return graphs

def making_dicts(exs,args):
    if not os.path.exists(args.dicts):
        attr_values=[]
        token_values=[]
        type_values=[]
        sep_values=[]
        identi_values=[]
        for ex_dict in exs:
            if ex_dict is None:
                continue

            for i in tqdm(ex_dict, ncols=150):
                text_src=i['text'][0]
                text_tgt=i['text'][1]
                for tk in text_src:
                    if isinstance(tk,str):
                        token_values.append(tk)
                    else:
                        token_values.append(str(tk.value))
                for tk in text_tgt:
                    if type(tk) in replace_type or tk.value in replace_type:
                        identi_values.append(tk.value)
                    elif type(tk) in not_replace_type:
                        if args.repeat_mode:
                            identi_values.append(tk.value)
                        sep_values.append(tk.value)
                    else:
                        print("error")
                        identi_values.append(tk.value)

                for tk_t in i ['tokens']:
                    attr_values.append(str(tk_t[1]))

                for tp in i ['types']:
                    type_values.append(tp[0])
                    attr_values.append(tp[1])

        logger.info("creating dicts")

        words_attr = load_words_pain(args, list(set(attr_values)))

        dictionary_attr = UnicodeCharsVocabulary(words_attr,
                                                args.max_characters_per_token,
                                                no_special_token=True)

        words_type = load_words_pain(args, list(set(type_values)))

        dictionary_type = UnicodeCharsVocabulary(words_type,
                                                args.max_characters_per_token,
                                                no_special_token=True)

        words_token = load_words_pain(args, token_values,args.src_vocab_size)

        dictionary_token = UnicodeCharsVocabulary(words_token,
                                                args.max_characters_per_token,
                                                no_special_token=False)

        words_sep = load_words_pain(args, list(set(sep_values)))

        dictionary_sep = UnicodeCharsVocabulary(words_sep,
                                                args.max_characters_per_token,
                                                no_special_token=False)

        words_identi = load_words_pain(args, identi_values,args.tgt_vocab_size)

        dictionary_identi = UnicodeCharsVocabulary(words_identi,
                                                args.max_characters_per_token,
                                                no_special_token=False)

        # in_counter=0
        # for tk in tqdm(token_values):
        #     if dictionary_token[tk] ==1:
        #         in_counter=in_counter+1
        #
        # OOV=in_counter/len(token_values)
        # print("OOV rate:{}".format(OOV))

        vocab_dicts={
            "dictionary_attr":dictionary_attr,
            "dictionary_type":dictionary_type,
            "dictionary_token":dictionary_token,
            "dictionary_sep":dictionary_sep,
            "dictionary_identi":dictionary_identi
        }

        pickle.dump(vocab_dicts,open(args.dicts,mode="wb"))
    else:
        vocab_dicts=pickle.load(open(args.dicts,mode="rb"))
        dictionary_attr=vocab_dicts['dictionary_attr']
        dictionary_type = vocab_dicts['dictionary_type']
        dictionary_token = vocab_dicts['dictionary_token']
        dictionary_sep = vocab_dicts['dictionary_sep']
        dictionary_identi = vocab_dicts['dictionary_identi']

    return dictionary_attr,dictionary_type,dictionary_token,dictionary_sep,dictionary_identi


def making_dicts_BPE(exs,args):
    if not os.path.exists(args.dicts):
        type_values=[]
        code_list = []

        for ex_dict in exs:
            if ex_dict is None:
                continue

            for i in tqdm(ex_dict, ncols=150):
                text_src=i['text'][0]
                text_tgt=i['text'][1]
                code_list.append(text_src)
                code_list.append(text_tgt)

                for tk_t in i ['tokens']:
                    code_list.append(str(tk_t))

                for tp in i ['types']:
                    type_values.append(tp)
        code_list = [i.encode('UTF-8', 'ignore').decode('UTF-8') for i in code_list]
        logger.info("creating dicts")
        bpe_tk = BPETokenizer(unk_token=UNK_WORD,pad_token=PAD_WORD,mask_token=MASK_WORD,lowercase=False)
        bpe_tk.add_tokens([PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD, MASK_WORD,BOS_S_WORD,STRING_TAG,REF_BEGIN_WORD,REF_END_WORD])
        bpe_tk.train_from_iterator(code_list, vocab_size=args.src_vocab_size,
                                   special_tokens=[PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD, MASK_WORD,BOS_S_WORD,STRING_TAG])
        dictionary_token=bpe_tk
        dictionary_identi = None


        dictionary_attr = ""

        words_type = load_words_pain(args, list(set(type_values)))

        dictionary_type = UnicodeCharsVocabulary(words_type,
                                                args.max_characters_per_token,
                                                no_special_token=True)



        vocab_dicts={
            "dictionary_attr":dictionary_attr,
            "dictionary_type":dictionary_type,
            "dictionary_token":bpe_tk,
            "dictionary_sep":"dictionary_sep"
        }

        pickle.dump(vocab_dicts,open(args.dicts,mode="wb"))
    else:
        vocab_dicts=pickle.load(open(args.dicts,mode="rb"))
        dictionary_attr=vocab_dicts['dictionary_attr']
        dictionary_type = vocab_dicts['dictionary_type']
        dictionary_token = vocab_dicts['dictionary_token']
        dictionary_sep = vocab_dicts['dictionary_sep']
        dictionary_identi = None


    constants.BOS = dictionary_token.token_to_id(BOS_WORD)
    constants.PAD = dictionary_token.token_to_id(PAD_WORD)
    constants.UNK = dictionary_token.token_to_id(UNK_WORD)
    constants.EOS = dictionary_token.token_to_id(EOS_WORD)
    constants.MASK = dictionary_token.token_to_id(MASK_WORD)
    constants.BOS_S = dictionary_token.token_to_id(BOS_S_WORD)
    constants.STRING_TAG_id = dictionary_token.token_to_id(STRING_TAG)
    constants.REF_END_id = dictionary_token.token_to_id(REF_END_WORD)
    constants.REF_BEGIN_id = dictionary_token.token_to_id(REF_BEGIN_WORD)



    return dictionary_attr,dictionary_type,dictionary_token,None,dictionary_identi


def get_node_representation(ex,vocab_token,vocab_type,vocab_attr,pretrain=False):
    if len(ex["tokens"]) > 500:
        ex["tokens"]=ex["tokens"][:500]
    if len(ex["types"]) > 500:
        ex["types"] = ex["types"][:500]


    node_len = len(ex["tokens"]) + len(ex["types"])
    edge_dicts = {}
    for key in edge_type.keys():
        if key in ex['edges'].keys():
            es = ex['edges'][key]
        else:
            es = []
        edge_metrix = torch.zeros([node_len, node_len])
        if len(es) > 0:
            index_edges = torch.tensor([e for e in es], dtype=torch.long).T
            index_edges = torch.where(index_edges > node_len - 1, node_len - 1, index_edges)
            edge_metrix = edge_metrix.index_put((index_edges[0], index_edges[1]), torch.ones(index_edges.shape[1]))
        edge_dicts[key] = edge_metrix

    tks=ex['tokens']
    tk_emb=[vocab_token.encode(i).ids for i in tks]
    len_of_sub_token=[len(i) for i in tk_emb]
    max_len_of_sub=max(len_of_sub_token)
    if pretrain:
        max_number=2
    else:
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


    node_types = torch.tensor([vocab_type[i] for i in ex['types']],
                              dtype=torch.int64)
    mark_range = torch.ones(node_len,dtype=torch.int64)
    mark_range_index=torch.tensor(ex['mark_range'],dtype=torch.int64)
    mark_range_index = torch.where(mark_range_index > node_len - 1, node_len - 1, mark_range_index)
    mark_range=mark_range.scatter(0,mark_range_index,2)

    return node_tokens,node_types,None,edge_dicts,mark_range






