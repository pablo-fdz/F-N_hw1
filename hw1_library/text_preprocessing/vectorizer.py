import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Union  # Allows setting as inputs of a function a set of options

def vectorizer(cv: Union[CountVectorizer, TfidfVectorizer], df: pd.DataFrame, column_text: str) -> pd.DataFrame:

    """
    Create a document-term matrix (DTM) using CountVectorizer or TfidfVectorizer.
    Args:
        cv (Union[CountVectorizer, TfidfVectorizer]): The vectorizer to use (CountVectorizer or TfidfVectorizer).
        df (pd.DataFrame): The DataFrame containing the text data.
        column_text (str): The name of the column containing the text data.
    Returns:
        dtm_dense (pd.DataFrame): The dense representation of the document-term matrix.
        terms (list): The list of terms extracted from the text data.
    """

    # Note that we can fit the count vectorizer with a pandas series
    cv.fit(df[column_text])
    dtm = cv.transform(df[column_text])  # Create DTM

    # Return dense interpretation of sparse matrix
    dtm_dense = dtm.todense()

    # Print DTM size
    print("Document-term matrix has size (documents, terms)", dtm_dense.shape)

    # Save extracted terms
    terms = cv.get_feature_names_out()

    return dtm_dense, terms