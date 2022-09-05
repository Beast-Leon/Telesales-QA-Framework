import string
import regex as re

def text_preprocessing(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    text = text.lower()
    return text
    