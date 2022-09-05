import numpy as np
import sys
sys.path.insert(0, "/Users/leon/Income/python files/Telesales-QA-Framework")
from helper.lexicons import *

def construct_sentence_vector(sentence, model, transformer_based = True):
    if transformer_based:
        return model.encode(sentence)
    else:   
        return np.array([token.vector for token in spacy_model(sentence)]).mean(axis=0)

def construct_dim_vector(descriptive_words, model):
    return np.array([construct_sentence_vector(sentence, model) for sentence in descriptive_words]).mean(axis=0)

def euclidian_distance(X, Y):
    return np.sqrt(np.sum(np.power(X-Y, 2)))

def cosine_sim(X, Y):
    return np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y))


# # if the max similarity < similarity_threshold, it means target categorys not found, then return ("no matching", input_sentence)
# # Otherwise, return the (matching class, input_sentence)
# def match_category(input_sentence, model, lexicon_type = "greeting",
#                   similarity_threshold = 0.6):
#     if lexicon_type == "greeting":
#         dic = greeting_lexicons
#     elif lexicon_type == "ending":
#         dic = ending_lexicons
#     new_vector = construct_sentence_vector(input_sentence, model)
#     similarity_ls = []
#     classes = []
#     for aspect, descriptive_words in dic.items():
#         classes.append(aspect)
#         cur_vector = construct_dim_vector(descriptive_words, model)
#         cur_similarity = cosine_sim(new_vector, cur_vector)
#         similarity_ls.append(cur_similarity)
#     max_similarity = max(similarity_ls)
#     if max_similarity < similarity_threshold:
#         return ("no_matching", input_sentence)
#     result_classes = classes[similarity_ls.index(max_similarity)]
#     return [input_sentence, result_classes]

# each sentence can be mapped to multiple categories according to the sim_threshold_dict
# return input sentence, resulting labels, class names, current similarity list
def match_multi_categories(input_sentence, model, sim_threshold_dict, lexicon_dict):
    # make sure the keys in sim_threshold_dict are in the same order with lexicon_dict
    assert sim_threshold_dict.keys() == lexicon_dict.keys()
    new_vector = construct_sentence_vector(input_sentence, model) # new input sentence vector
    similarity_ls = []
    classes = []
    # calculate the average cosine similarity of the current new input sentence vector for each dictionary category
    for aspect, descriptive_words in lexicon_dict.items():
        classes.append(aspect)
        cur_vector = construct_dim_vector(descriptive_words, model)
        cur_similarity = cosine_sim(new_vector, cur_vector)
        similarity_ls.append(cur_similarity)
    # Do a comparison of the generated similarity_ls with the ground-truth similarity list
    threshold_ls = list(sim_threshold_dict.values())
    similar_label_idx = [index for index, (similarity, threshold) in enumerate(zip(similarity_ls, threshold_ls)) if similarity >= threshold] 
    if not similar_label_idx:
        return [input_sentence, ["no matching"]], classes, similarity_ls
    result_labels = [classes[i] for i in similar_label_idx]
    return [input_sentence, result_labels], classes, similarity_ls


# def batch_match_category(sentence_ls, model, lexicon_type = "opening", similarity_threshold = 0.6):
#     result_ls = []
#     for sentence in sentence_ls:
#         result_ls.append(match_category(sentence, model, lexicon_type, similarity_threshold))
#     return result_ls

def batch_match_multi_categories(sentence_ls, model, sim_threshold_dict, lexicon_dict):
    result_ls = []
    label_ls = []
    similarity_ls = []
    for sentence in sentence_ls:
        result, classes, similarity = match_multi_categories(sentence, model, sim_threshold_dict, lexicon_dict)
        result_ls.append(result)
        label_ls.append(result[1])
        similarity_ls.append(similarity)
    return result_ls, sentence_ls, label_ls, similarity_ls

# # Assume match_result_ls has length > 1
# # Cluster contigious sentences with same category together
# def cluster_category(match_result_ls):
#     sen_store = match_result_ls[0][0]
#     temp_label = match_result_ls[0][1]
#     new_result_ls = []
#     for i in range(1, len(match_result_ls)):
#         cur_label = match_result_ls[i][1]
#         cur_sen = match_result_ls[i][0]
#         if cur_label == temp_label:
#             sen_store += " " + cur_sen
#         else:
#             new_result_ls.append([sen_store, temp_label])
#             sen_store = cur_sen
#             temp_label = cur_label
#     if sen_store:
#         new_result_ls.append([sen_store, temp_label])
#     return new_result_ls
        
# # result_ls is the result of batch_match_category function
# # bool_group = True: group sentences with same label together
# def nlp_aspect_matching(sentence_ls, model, lexicon_type = "greeting", bool_group = True, similarity_threshold = 0.4):
#     result_ls = batch_match_category(sentence_ls, model, lexicon_type, similarity_threshold) # generate label for each sentence
#     if not bool_group or len(result_ls) == 1: # if the user don't want to cluster same label sentences, just return the result_ls,     # Or if the result_ls only contains one sentence, just return it
#         return result_ls
#     else: # If there are more than 1 sentence in the list and bool_group = True
#         new_result_ls = cluster_category(result_ls)
#         return new_result_ls
            

    
        