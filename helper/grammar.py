grammars = r""" 
    JP: {<JJ.*>}
    NP: {<JP|CD>*<PRP.*|DT|NN.*>+}
    PP: {<IN|TO|RP><NP|VB.*>} 
    VP: {<VB.*|RB.*>+<PP|NP>*}
    Sentence: {<UH>*<JP|NP>*<MD|IN>*<VP|PP|NP|JP>+}
    Question: {<MD|WDT|DP|WRB|><MD>*<Sentence|NP|PP|VP|JP>}
"""