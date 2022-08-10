import nltk
import spacy

pos_dict = {'ocbc_plus': {'pos': 'NOUN', 'tag': 'NN'}}



def pos_postprocessor_pipe(doc) :
    for token in doc :
        text = token.text
        if text in pos_dict:
            token.pos_ = pos_dict[text]['pos']
            token.tag_ = pos_dict[text]['tag']
    return doc