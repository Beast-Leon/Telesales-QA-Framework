import nltk
import spacy

pos_dict = {'charter_plus': {'pos': 'NOUN', 'tag': 'NN'},
            'free': {'pos': 'ADJ', 'tag': 'JJ'},
           'one_or_two': {'pos': 'NUM', 'tag': 'CD'},
           'single_or_married': {'pos': 'ADJ', 'tag': 'JJ'}}
# pattern_ls = [{"pattern": [{"LOWER": "charter"}, {"LOWER": "plus"}], "attr": {"TAG": "NN", "POS": "NOUN"}}]
# # spacy --version 2.3.5
# def pos_postprocessor_pipe(doc) :
#     for token in doc :
#         text = token.text
#         if text in pos_dict:
#             token.pos_ = pos_dict[text]['pos']
#             token.tag_ = pos_dict[text]['tag']
#     return doc

# spacy --version > 3.0
from spacy.language import Language

@Language.component("pos_postprocessor_pipe")
def pos_postprocessor_pipe(doc) :
    for token in doc :
        text = token.text
        if text in pos_dict:
            token.pos_ = pos_dict[text]['pos']
            token.tag_ = pos_dict[text]['tag']
    return doc

pos_ls = [{"pattern": [[{"LOWER": "charter"}, {"LOWER": "plus"}]], "attr": {"TAG": "NN", "POS": "NOUN"}},
          {"pattern": [[{"LOWER": "free"}]], "attr": {"TAG": "JJ", "POS": "ADJ"}},
          {"pattern": [[{"LOWER": "one"}, {"LOWER": "or"}, {"LOWER": "two"}]], "attr": {"TAG": "CD", "POS": "NUM"}},
          {"pattern": [[{"LOWER": "single"}, {"LOWER": "or"}, {"LOWER": "married"}]], "attr": {"TAG": "JJ", "POS": "ADJ"}}
         ]

def add_pos(cust_rules, model):
    ruler = model.get_pipe('attribute_ruler')
    for pos in pos_ls:
        index_len = len(pos['pattern'][0])
        for i in range(index_len):
            ruler.add(patterns = pos['pattern'], attrs = pos['attr'], index = i)
    