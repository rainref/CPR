PAD = 0
UNK = 1
BOS = 2
EOS = 3
CLS = 4
MASK=4
BOS_S=5
REF_END_id=-1
REF_BEGIN_id=-1

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
CLS_WORD = "<cls>"
REF_BEGIN_WORD = "<ref>"
REF_END_WORD="</ref>"

MASK_WORD = "MASK"
BOS_S_WORD="<s_no_graph>"
STRING_TAG="___"


edge_type = {'Child': 0, 'NextToken': 1, 'NextUse': 2,"InField":3}
edge_color = {'Child': 'yellow', 'NextToken': 'blue', 'Subtoken': 'green', 'LastLexicalUse': 'red'}
#edge_type = {'Child': 0, 'Subtoken': 1, 'LastLexicalUse': 2}


NODE_TYPE_MAP={
    "AST":0,
    "Token":1,
    "Subtoken":2,
    "PAD":3
}
NODE_TYPE_SIZE=len(NODE_TYPE_MAP)

TOKEN_TYPE_MAP = {
    # Java
    '<pad>': 0,
    '<unk>': 1,
    'other': 2,
    'var': 3,
    'method': 4,
    # Python
    's': 5,
    'None': 6,
    'value': 7,
    'asname': 8,
    'n': 9,
    'level': 10,
    'is_async': 11,
    'arg': 12,
    'attr': 13,
    'id': 14,
    'name': 15,
    'module': 16
}

AST_TYPE_MAP = {
    '<pad>': 0,
    'N': 1,
    'T': 2
}

DATA_LANG_MAP = {
    'java': 'java',
    'python': 'python',
    "java_bpe":"java",
    "java_bpe_simple":"java"
}

LANG_ID_MAP = {
    'java': 0,
    'python': 1,
    'c#': 2,
    "java_bpe":0,
    "java_bpe_simple":0
}
