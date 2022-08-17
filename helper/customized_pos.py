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