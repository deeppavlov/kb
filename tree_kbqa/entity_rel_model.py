from collections import Counter
from ufal_udpipe import Model as udModel, Pipeline
from udapi.block.read.conllu import Conllu
from io import StringIO
import nltk

def descendents(node, desc_list):
    if len(node.children) > 0:
        for child in node.children:
            desc_list = descendents(child, desc_list)
    desc_list.append(node.form)

    return desc_list


class Parser:
    def __init__(self, udpipe_path):
        self.ud_model = udModel.load(udpipe_path)
        self.full_ud_model = Pipeline(self.ud_model, "vertical", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

    def __call__(self, sentence):
        q_tokens = nltk.word_tokenize(sentence)
        q_str = '\n'.join(q_tokens)
        s = self.full_ud_model.process(q_str)
        tree = Conllu(filehandle=StringIO(s)).read_tree()
        
        fnd = False
        fnd, detected_entity, detected_rel = self.find_entity(tree, q_tokens)
        if fnd == False:
            fnd, detected_entity, detected_rel = self.find_entity_adj(tree)

        return detected_entity, detected_rel

    def find_entity(self, tree, q_tokens):
        detected_entity = ""
        detected_rel = ""
        for node in tree.descendants:
            if len(node.children) <= 2 and node.upos in ["NOUN", "PROPN"]:
                desc_list = []
                entity_tokens = []
                while node.parent.upos in ["NOUN", "PROPN"] and node.parent.deprel!="root":
                    node = node.parent
                detected_rel = node.parent.form
                desc_list.append(node.form)
                desc_list = descendents(node, desc_list)
                num_tok = 0
                for n, tok in enumerate(q_tokens):
                    if tok in desc_list:
                        entity_tokens.append(tok)
                        num_tok = n
                if q_tokens[(num_tok+1)].isdigit():
                    entity_tokens.append(q_tokens[(num_tok+1)])
                detected_entity = ' '.join(entity_tokens)
                return True, detected_entity, detected_rel

        return False, detected_entity, detected_rel

    def find_entity_adj(self, tree):
        detected_rel = ""
        detected_entity = ""
        for node in tree.descendants:
            if len(node.children) <= 1 and node.upos == "ADJ":
                detected_rel = node.parent.form
                detected_entity = node.form
                return True, detected_entity, detected_rel
        
        return False, detected_entity, detected_rel
        

prs = Parser("russian-syntagrus-ud-2.3-181115.udpipe")
sentence = "Когда был изобретен автомат Калашникова?"
detected_entity, detected_rel = prs(sentence)
print(detected_entity, detected_rel)


