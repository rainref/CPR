import numpy as np
from component.inputters.vocabulary import Vocabulary
import networkx as nx
import matplotlib.pyplot as plt
from component.inputters import constants
import heapq


class Graph(object):
    def __init__(self, dict=None,_id=None):
        self._id = 0
        self._text = None
        self._edges = []
        self._tokens = []
        self._tokens_length = None
        self._backbone_sequence = []
        self.src_vocab = Vocabulary(no_special_token=True)  # required for Copy Attention
        self.code_tokens = []
        self.type_sequence = None
        self.G = None
        self.pos = None
        if dict is not None:
            self._edges = dict['edges']

            self._tokens = dict['node_labels']

    @property
    def id(self) -> str:
        return self._id

    @property
    def text(self) -> str:
        return self._text

    @property
    def tokens_length(self) -> str:
        return self._tokens_length

    @text.setter
    def text(self, param: str) -> None:
        self._text = param

    @property
    def edges(self) -> list:
        return self._edges

    @edges.setter
    def edges(self, param: list) -> None:
        assert isinstance(param, list)
        self._edges = param

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._tokens_length = len(param)
        self._tokens = param

    def form_src_vocab(self) -> None:
        for i in range(len(self.backbone_sequence)):
            token = self.tokens[self.backbone_sequence[i]]
            self.src_vocab.add(token)
            self.code_tokens.append(token)

    @property
    def backbone_sequence(self) -> list:
        return self._backbone_sequence

    @backbone_sequence.setter
    def backbone_sequence(self, param: list) -> None:
        assert isinstance(param, list)
        for i in range(len(param)):
            token = self.tokens[param[i]]
            if token == "":  # for python dataset
                self.tokens[param[i]] = "pythonblank"
                token = "pythonblank"
            self.src_vocab.add(token)
            self.code_tokens.append(token)
        self._backbone_sequence = param

    def vectorize(self, word_dict, _type='word') -> list:
        if _type == 'word':
            return [word_dict[w] for w in self.tokens]
        elif _type == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.tokens]
        else:
            assert False

    def code_vectorize(self, word_dict, _type='word') -> list:
        if _type == 'word':
            return [word_dict[w] for w in self.code_tokens]
        elif _type == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.code_tokens]
        else:
            assert False

    def getmatrix(self, edge_type):
        matrix = np.zeros((self._tokens_length, self._tokens_length * len(edge_type) * 2))
        for edge in self.edges:
            if edge[0] in edge_type:
                e_type = edge_type[edge[0]]
                src_idx = edge[1]
                tgt_idx = edge[2]
                matrix[tgt_idx][e_type * self._tokens_length + src_idx] = 1
                matrix[src_idx][(e_type + len(edge_type)) * self._tokens_length + tgt_idx] = 1
        return matrix

    def showGNN(self, layout=None):
        gnn = self
        G = gnn.G
        if G is None:
            G = nx.Graph()
            node_tuple = []
            for index, tk in enumerate(gnn.tokens):
                node_tuple.append((index, {'token': tk}))
            G.add_nodes_from(node_tuple)

            edge_tuple = []
            for index, edge in enumerate(gnn.edges):
                edge_tuple.append((edge[1], edge[2], {'type': edge[0]}))
            G.add_edges_from(edge_tuple)
            self.G = G
        if layout == None:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = layout

        label = [target for target in G.nodes.data('token')]
        tags = {n: lab for (n, lab) in label if n in pos and (len(lab) < 8 or lab=="mask_code")}
        nx.draw_networkx_nodes(G, pos=pos, node_size=100)
        nx.draw_networkx_labels(G, pos=pos, labels=tags, font_size=9)

        edge_color = constants.edge_color
        for key, value in edge_color.items():
            tarList = [target for target in G.edges.data('type') if target[2] == key]
            nx.draw_networkx_edges(G, pos=pos, edgelist=tarList, edge_color=value)
        plt.show()
        return pos


    def showGNN_attribute(self, layout=None,attribute=None,n = 10):
        gnn = self
        G = gnn.G
        attribute=attribute[:len(gnn.tokens)]
        if G is None:
            G = nx.Graph()
            node_tuple = []
            for index, tk in enumerate(gnn.tokens):
                node_tuple.append((index, {'token': tk}))
            G.add_nodes_from(node_tuple)

            edge_tuple = []
            for index, edge in enumerate(gnn.edges):
                edge_tuple.append((edge[1], edge[2], {'type': edge[0]}))
            G.add_edges_from(edge_tuple)
            self.G = G
        if layout == None:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = layout

        label = [target for target in G.nodes.data('token')]

        vmin=attribute.min()
        vmax=attribute.max()
        nx.draw_networkx_nodes(G, pos=pos, node_size=100,cmap='cool',node_color=attribute,vmin=vmin,vmax=vmax)


        a = attribute
        max_indexs = heapq.nlargest(n, range(len(a)), a.take)

        tags = {n: lab for (n, lab) in label if n in pos and n in max_indexs}
        for n,lab in tags.items():
            if len(lab)>8:
                tags[n]=lab[:7]+".."
        nx.draw_networkx_labels(G, pos=pos, labels=tags, font_size=9)

        edge_color = constants.edge_color
        for key, value in edge_color.items():
            tarList = [target for target in G.edges.data('type') if target[2] == key]
            nx.draw_networkx_edges(G, pos=pos, edgelist=tarList, edge_color=value)
        plt.show()
        return pos