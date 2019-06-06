import nltk
import fastText
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ufal_udpipe import Model as udModel, Pipeline
from udapi.block.read.conllu import Conllu
from io import StringIO

fasttext_load_path = "/home/dmitry/.deeppavlov/downloads/embeddings/lenta_lower_100.bin"
udpipe_load_path = "russian-syntagrus-ud-2.3-181115.udpipe"
sentence = "Когда был изобретен автомат Калашникова?"
entity = "автомат Калашникова"

class SentenceGraph:
    def __init__(self, fasttext_load_path, udpipe_load_path):
        self.fasttext = fastText.load_model(str(fasttext_load_path))
        ud_model = udModel.load(udpipe_load_path)
        self.full_ud_model = Pipeline(ud_model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

    def __call__(self, sentence, entity):
        s_tokens = nltk.word_tokenize(sentence)
        e_tokens = nltk.word_tokenize(entity)
        s = self.full_ud_model.process(sentence)
        tree = Conllu(filehandle=StringIO(s)).read_tree()

        g = dgl.DGLGraph()
        g.add_nodes(len(s_tokens))
        g = self.add_features(s_tokens, g)
        g = self.add_edges(tree, g)
        labels = self.make_labels(s_tokens, e_tokens)
        return g, labels

    def add_features(self, s_tokens, g):
        emb_list = []
        for tok in s_tokens:
            emb_list.append(self.fasttext.get_word_vector(tok))
        emb_tensor = th.tensor(emb_list)
        g.ndata['x'] = emb_tensor
        return g

    def add_edges(self, tree, g):
        for node in tree.descendants:
            ord_1 = node.ord
            for child in node.children:
                ord_2 = child.ord
                g.add_edge(ord_1, ord_2)
        return g

    def make_labels(self, s_tokens, e_tokens):
        labels = []
        for tok in s_tokens:
            if tok in e_tokens:
                labels.append([1.0, 0.0])
            else:
                labels.append([0.0, 1.0])
        return labels

sg = SentenceGraph

msg = fn.copy_src(src = 'h', out = 'm')

def reduce(nodes):
    accum = th.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        
    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        
    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, 2, F.relu)])

    def forward(self, g):
        for conv in self.layers:
            h = conv(g, h)
            
        return h

model = Classifier(100, 200)


