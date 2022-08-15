import numpy as np

# classes with respective sample sentences
greeting_lexicons = {'opening': ['good time to talk', 'good morning', 'good afternoon', 'good evening', 'calling from income', 'hello i am looking for mister', 'may i speak with', 'calling for the number'],
                    'purpose_of_call': ['the purpose of my call is to inform you', 'as a valued customer of income', 'we have designed this plan called', 'this plan is for exclusive customers like yourself', 'we formulated a special fiftieth anniversary insurance bundle called the i fifty', 'we have this anniversary plan for the family'],
                    'ask_for_permission': ['how may i address you', 'may i call you', 'is it a good time to talk to you', 'are you convenient to talk']}


def construct_sentence_vector(sentence, spacy_model):
    return np.array([token.vector for token in spacy_model(sentence)]).mean(axis=0)

def construct_dim_vector(descriptive_words, spacy_model):
    return np.array([construct_sentence_vector(sentence, spacy_model) for sentence in descriptive_words]).mean(axis=0)

def euclidian_distance(X, Y):
    return np.sqrt(np.sum(np.power(X-Y, 2)))

def cosine_sim(X, Y):
    return np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y))

# if the max similarity < similarity_threshold, it means target categorys not found, then return ("no matching", input_sentence)
# Otherwise, return the (matching class, input_sentence)
def match_category(input_sentence, spacy_model, lexicon_type = "greeting",
                  similarity_threshold = 0.6):
    if lexicon_type == "greeting":
        dic = greeting_lexicons
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
        return ("no matching", input_sentence)
    result_classes = classes[similarity_ls.index(max_similarity)]
    return [result_classes, input_sentence]
       
def batch_match_category(sentence_ls, spacy_model, lexicon_type = "greeting"):
    result_ls = []
    for sentence in sentence_ls:
        result_ls.append(match_category(sentence, spacy_model, lexicon_type))
    return result_ls

# result_ls is the result of batch_match_category function
# bool_group = True: group sentences with same label together
# Notice: bool_group = True will cause error, need to debug
def nlp_aspect_matching(sentence_ls, spacy_model, lexicon_type = "greeting", bool_group = True):
    result_ls = batch_match_category(sentence_ls, spacy_model, 'greeting') # generate label for each sentence
    if not bool_group: # if the user don't want to cluster same label sentences, just return the result_ls
        return result_ls
    elif len(result_ls) == 1: # Or if the result_ls only contains one sentence, just return it
        return result_ls
    else:
        sen_store = "" # initialize the accumulated sentence
        new_result_ls = []
        temp_label = "" # initialize the temprorary label
        for i in range(0, len(result_ls)): # Loop through the sentence list
            cur_label = result_ls[i][0]
#             if cur_label == "no matching":
#                 new_result_ls.append(result_ls[i])
#                 continue
            if cur_label != temp_label: # if current label is different than the temprorary label, we either append the current sentence or append the accumulated sen_store
                if not sen_store:
                    sen_store += " " + result_ls[i][1]
                    temp_label = cur_label
                    #new_result_ls.append([cur_label, sen_store])
                else:
                    new_result_ls.append([temp_label, sen_store])
                    temp_label = cur_label
                    sen_store = ""
            else: # if current label is same as the temprorary label, append the current sentence to the sen_store
                sen_store += " " + result_ls[i][1]
                continue
        if sen_store: # if the sen_store contain sentences after the loop, append to the new_result_ls
            new_result_ls.append([temp_label, sen_store])
        return new_result_ls
            

    
        