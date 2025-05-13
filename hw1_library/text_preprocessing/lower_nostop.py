from nltk.tokenize import word_tokenize  # For tokenizing
import re

def lower_nostop(text, rm_stopwords = False, stopword_set = None):
    """
    Preprocess text by:
       - Converting to lowercase.
       - Removing punctuation and digits.
       - Tokenizing.
       - Removing stopwords (optional).
    
    Returns:
        list: A list of tokens lowercased and without punctuation.
    """
    text_lower = text.lower()
    
    text_no_punct = re.sub(r'[^\w\s]', ' ', text_lower)  # Remove punctuation and replace with whitespace
    tokens = word_tokenize(text_no_punct)
    tokens = word_tokenize(text_no_punct)
    # Remove stopwords if desired
    if rm_stopwords == True:
        tokens = [token for token in tokens if token not in stopword_set]
    # We return the whole string of tokens so that we can find n-grams later
    return " ".join(tokens)