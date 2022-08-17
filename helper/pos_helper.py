import nltk
import spacy
from customized_pos import *
#nlp = spacy.load('en_core_web_lg')

# grammar is some grammar rules like r""" NP: {<PRP.*|DT|JJ|NN.*>+}"""
def parse(spacy_doc, grammar):
    pos_ls = []
    for token in spacy_doc:
        pos_ls.append((token.text, token.tag_))
    chunkParser = nltk.RegexpParser(grammar)
    chunked = chunkParser.parse(pos_ls)
    return chunked

# Traverse through the parse tree, and return (each sentence, label)
def run_traverse(t):
    new_sens = []
    labels = []
    level = 0
    def traverse(t, level = 0, ls = []):
        try:
            if t.label() and level == 1 and ls:
                new_sens.append(ls)
                ls = []
            if t.label() and level == 1:
                labels.append(t.label())
        except AttributeError:
            ls.append(t[0])
        else:
            for i, child in enumerate(t):
                if i == 0:
                    level += 1
                ls = traverse(child, level, ls)
                if i == len(t)-1:
                    level -= 1
        return ls
    ls = traverse(t, level)
    new_sens.append(ls)
    return new_sens, labels

# Group each label with each sentence as a tuple, and group sentences with less 5 words with the next sentence together
# parts is the return of run_traverse
# input example(sentences, labels)
# sen_return_type can be either "list" or "string"
def cluster_part(parts, cluster_threshold = 7, sen_return_type = "string"):
    sen_return_type = sen_return_type.lower()
    if sen_return_type != "string" and sen_return_type != "list":
        raise Exception(f"Current return type {sen_return_type} is not supported.")
    new_sen = []
    temp_sen = []
    # If false, means we won't store current sentence to the temporary list, Otherwise, store the current sentence
    # and insert to the front of the next sentence
    activated = False
    for sentence, label in zip(parts[0], parts[1]):
        if len(sentence) < cluster_threshold and len(temp_sen) < cluster_threshold:
            activated = True
            temp_sen.extend(sentence)
            continue
        else:
            activated = False
            if temp_sen:
                sentence = temp_sen + sentence
                temp_sen = []
            if sen_return_type == "string":
                sentence = " ".join([i for i in sentence])
            new_sen.append([label, sentence])       
    return new_sen

# Combine the above functions together
def nlp_sentencizer(transcript, grammar, spacy_model):
    doc = spacy_model(transcript)
    tree = parse(doc, grammar)
    parts = run_traverse(tree)
    result = cluster_part(parts)
    return result