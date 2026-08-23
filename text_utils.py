import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()
_stop_words = set(ENGLISH_STOP_WORDS)

def preprocess(text: str) -> list:
    """
    Standardized text preprocessing pipeline for Information Retrieval.
    - Lowercase
    - Tokenization (alphanumeric words)
    - Stop-word removal
    - Porter Stemming
    """
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    words = re.findall(r'\b[a-z0-9]+\b', text)
    
    processed_tokens = []
    for word in words:
        if word not in _stop_words and len(word) > 2:
            stemmed = _stemmer.stem(word)
            processed_tokens.append(stemmed)
            
    return processed_tokens

def identity_tokenizer(text):
    """
    Dummy tokenizer for TF-IDF when we pass a list of tokens directly.
    """
    return text
