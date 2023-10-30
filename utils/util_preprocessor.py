import re
import html2text
from typing import Any
import numpy as np

# from spacy.lang.en import English
# import spacy

# from nltk.tokenize import sent_tokenize
# import nltk
# nltk.download('punkt')

def add_subject_to_body(
    subject: str,
    body: str
) -> str:
    """Add subject to body

    Args:
        subject (str): subject of email
        body (str): body of email

    Returns:
        str: subject + body
    """
    
    if subject in body:
        return body
    else:
        return subject + ' ' + body

def convert_string_to_list(
    text: str,
    sep: str = ',',
    preprocessor: Any = None,
) -> list:
    """Convert string to list -> useuful when string has a lot of different elements separated by a specific character

    Args:
        text (str): text to convert to list
        sep (str, optional): character to separate elements in text. Defaults to ','.
        preprocessor (Any, optional): Preprocessor to preprocess text. Defaults to None.

    Returns:
        list: list of strings
    """

    text = text.split(sep)

    # if preprocessor is None:
    #     preprocessor = Preprocessor()
    
    # text = [preprocessor(val) for val in text]
    
    return [item.strip() for item in text]