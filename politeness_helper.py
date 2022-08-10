import nltk
import spacy
#nlp = spacy.load('en_core_web_lg')
import pickle

from politeness.scripts.format_input import *
from politeness.scripts.train_model import *
from politeness.features.vectorizer import PolitenessFeatureVectorizer

# Load standardizer
# with open('other_collection/standardizer.pkl', 'rb') as f:
#     standardizer = pickle.load(f)

# Get parses using spacy, then we need the parse to pass in the politeness model.
def get_parses(spacy_doc):
    parse = {'deps': [], 'sent': ""}
    try:
        parse_ls = []
        for token in spacy_doc:
            parse_str = ""
            cur_dep = token.dep_
            head_text = token.head.text
            cur_text = token.text
            if cur_dep == 'ROOT':
                cur_dep = cur_dep.lower()
                head_id = 0
                head_text = 'ROOT'
                cur_id = token.i + 1
            else:
                head_id = token.head.i + 1
                cur_id = token.i + 1
            parse_str += cur_dep + "(" + head_text + "-" + str(head_id) + ", " + cur_text + "-" + str(cur_id) + ")"
            parse_ls.append(parse_str)
        parse['deps'] = parse_ls
        parse['sent'] = spacy_doc.text
    except Exception as e:
        print(e)
    return parse

# If format_doc for training, need to pass score.
def format_doc(doc, spacy_model, score = None):
    sents = get_sentences(doc)
    raw_parses = []
    for sent in sents:
        nlp_sent = spacy_model(sent)
        raw_parses.append(get_parses(nlp_sent))
    if score == None:
        parse_dict = {"text": doc,
            "sentences": [],
            "parses": []}
    else:
        parse_dict = {"text": doc,
                    "sentences": [],
                    "parses": [],
                    "score": score}
    for raw in raw_parses:
        parse_dict['sentences'].append(clean_treeparse(raw['sent']))
        parse_dict['parses'].append(raw['deps'])
    return parse_dict

def customize_score(request, clf, standardizer, score_format = "int", pred_threshold = 0.5):
    """
    :param request - The request document to score
    :type request - dict with 'sentences' and 'parses' field
        sample (taken from test_documents.py)--
        {
            'sentences': [
                "Have you found the answer for your question?",
                "If yes would you please share it?"
            ],
            'parses': [
                ["csubj(found-3, Have-1)", "dobj(Have-1, you-2)",
                 "root(ROOT-0, found-3)", "det(answer-5, the-4)",
                 "dobj(found-3, answer-5)", "poss(question-8, your-7)",
                 "prep_for(found-3, question-8)"],
                ["prep_if(would-3, yes-2)", "root(ROOT-0, would-3)",
                 "nsubj(would-3, you-4)", "ccomp(would-3, please-5)",
                 "nsubj(it-7, share-6)", "xcomp(please-5, it-7)"]
            ]
        }

    returns class probabilities as a dict
        { 'polite': float, 'impolite': float }
    """
    # Vectorizer returns {feature-name: value} dict
    vectorizer = PolitenessFeatureVectorizer()
    features = vectorizer.features(request)
    fv = [features[f] for f in sorted(features.keys())]
    # Single-row sparse matrix
    X = csr_matrix(np.asarray([fv]).astype("float"))
    X = standardizer.transform(X)
    if score_format == "int":
        probs = clf.predict_proba(X)
        probs_binary = (probs[:,1] >= pred_threshold).astype('int')
        return probs_binary[0]
    elif score_format == "prob":
        probs = clf.predict_proba(X)
        # Massage return format
        probs_dict = {"polite": probs[0][1], "impolite": probs[0][0]}
        return probs_dict
    
# category_sentence is a list with tuples, each tuple contains (category (like opening, purpose of call), sentence)
def nlp_politeness(category_sentence, clf, spacy_model, standardizer, score_format = "int", pred_threshold = 0.5):
    result_politeness = []
    for category, sentence in category_sentence:
        cur_parse = format_doc(sentence, spacy_model)
        cur_score = customize_score(cur_parse, clf, standardizer, score_format, pred_threshold)
        if cur_score == 1:
            politeness = "polite"
        elif cur_score == 0:
            politeness = "impolite"
        result_politeness.append((category, sentence, politeness))
    return result_politeness
