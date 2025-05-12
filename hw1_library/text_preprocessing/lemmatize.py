from nltk.stem import WordNetLemmatizer  # For lemmatizing

def lemmatize(text):
    """
    Preprocess text by applying lemmatized.
    Should just input a string which has been previously pre-processed, which at least removes
    the punctuation.

    Returns:
        str: A string of lemmatized tokens separated by spaces.
    """

    tokens = text.split()  # Split input text based on whitespaces
    lemmatizer = WordNetLemmatizer()  # Initiallize lemmatizer
    lemmatized_text = []  # Initialize empty list to store lemmatized text
    for word in tokens:
        lemmatized_text.append(lemmatizer.lemmatize(word))

    return " ".join(lemmatized_text)