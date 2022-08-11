import nltk
import spacy

pos_dict = {'charter_plus': {'pos': 'NOUN', 'tag': 'NN'},
            'free': {'pos': 'ADJ', 'tag': 'JJ'},
           'one_or_two': {'pos': 'NUM', 'tag': 'CD'},
           'single_or_married': {'pos': 'ADJ', 'tag': 'JJ'}}



def pos_postprocessor_pipe(doc) :
    for token in doc :
        text = token.text
        if text in pos_dict:
            token.pos_ = pos_dict[text]['pos']
            token.tag_ = pos_dict[text]['tag']
    return doc