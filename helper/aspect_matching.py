import numpy as np
import sys
sys.path.insert(0, "/Users/leon/Income/python files/politeness_code")
from helper.lexicons import *

def construct_sentence_vector(sentence, spacy_model):
    return np.array([token.vector for token in spacy_model(sentence)]).mean(axis=0)

def construct_dim_vector(descriptive_words, spacy_model):
    return np.array([construct_sentence_vector(sentence, spacy_model) for sentence in descriptive_words]).mean(axis=0)

def euclidian_distance(X, Y):
    return np.sqrt(np.sum(np.power(X-Y, 2)))

def cosine_sim(X, Y):
    return np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y))

# take similarity_pair (e.g. [0.85, 0.8, 0.75] for each category class)
# return index of all the category that within the threshold comparing to the max prob category
def similarity_comparison(similarity_pair, threshold = 0.05):
    pass
# if the max similarity < similarity_threshold, it means target categorys not found, then return ("no matching", input_sentence)
# Otherwise, return the (matching class, input_sentence)

def match_category(input_sentence, spacy_model, lexicon_type = "greeting",
                  similarity_threshold = 0.6):
    if lexicon_type == "greeting":
        dic = greeting_lexicons
    elif lexicon_type == "ending":
        dic = ending_lexicons
    new_vector = construct_sentence_vector(input_sentence, spacy_model)
    similarity_ls = []
    classes = []
    for aspect, descriptive_words in dic.items():
        classes.append(aspect)
        cur_vector = construct_dim_vector(descriptive_words, spacy_model)
        cur_similarity = cosine_sim(new_vector, cur_vector)
        similarity_ls.append(cur_similarity)
    max_similarity = max(similarity_ls)
    if max_similarity < similarity_threshold:
        return ("no_matching", input_sentence)
    result_classes = classes[similarity_ls.index(max_similarity)]
    return [result_classes, input_sentence]



def batch_match_category(sentence_ls, spacy_model, lexicon_type = "greeting"):
    result_ls = []
    for sentence in sentence_ls:
        result_ls.append(match_category(sentence, spacy_model, lexicon_type))
    return result_ls

# Assume match_result_ls has length > 1
# Cluster contigious sentences with same category together
def cluster_category(match_result_ls):
    sen_store = match_result_ls[0][1]
    temp_label = match_result_ls[0][0]
    new_result_ls = []
    for i in range(1, len(match_result_ls)):
        cur_label = match_result_ls[i][0]
        cur_sen = match_result_ls[i][1]
        if cur_label == temp_label:
            sen_store += " " + cur_sen
        else:
            new_result_ls.append([temp_label, sen_store])
            sen_store = cur_sen
            temp_label = cur_label
    if sen_store:
        new_result_ls.append([temp_label, sen_store])
    return new_result_ls
        
# result_ls is the result of batch_match_category function
# bool_group = True: group sentences with same label together
def nlp_aspect_matching(sentence_ls, spacy_model, lexicon_type = "greeting", bool_group = True):
    result_ls = batch_match_category(sentence_ls, spacy_model, lexicon_type) # generate label for each sentence
    if not bool_group or len(result_ls) == 1: # if the user don't want to cluster same label sentences, just return the result_ls,     # Or if the result_ls only contains one sentence, just return it
        return result_ls
    else: # If there are more than 1 sentence in the list and bool_group = True
        new_result_ls = cluster_category(result_ls)
        return new_result_ls
            

    
        