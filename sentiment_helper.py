import numpy as np

def softmax(x, axis):
    e_x = np.exp(x - np.max(x, axis))
    return e_x / (e_x.sum(axis) + 1e-9)

def batch_tokenizer(sentence_ls, tokenizer, sentiment_model):
    softmax_ls = []
    for sentence in sentence_ls:
        cur_input = tokenizer(sentence, return_tensors = "pt")
        output = sentiment_model(**cur_input).logits.detach().numpy()[0]
        soft_output = softmax(output, 0)
        softmax_ls.append(soft_output)
    return softmax_ls

# generate label
def generate_label(softmax_ls):
    label_ls = []
    label = ['negative', 'neutral', 'positive']
    for i, prob in enumerate(softmax_ls):
        index = np.argmax(prob)
        label_ls.append(label[index])
    return label_ls

# Combine the result of the aspect matching result with the sentiment label
def nlp_sentiment(aspect_ls, tokenizer, sentiment_model):
    sentence_ls = list(map(lambda x: x[1], aspect_ls))
    softmax_ls = batch_tokenizer(sentence_ls, tokenizer, sentiment_model)
    label_ls = generate_label(softmax_ls)
    assert len(aspect_ls) == len(label_ls)
    result_ls = []
    for aspect, label in zip(aspect_ls, label_ls):
        cur_result = aspect + [label]
        result_ls.append(cur_result)
    return result_ls
