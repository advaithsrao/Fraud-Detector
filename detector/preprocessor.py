import sys
sys.path.append("..")

import re
import html2text
from typing import Any
import numpy as np

from utils.util_preprocessor import add_subject_to_body

class Preprocessor:
    def __call__(
        self, 
        text: str,
    ) -> str:
        """Preprocess text

        Args:
            text (str): text to preprocess
        
        Returns:
            text (str): preprocessed text
        """

        text = self.remove_links(text)

        text = self.replace_html_content(text)

        text = self.remove_new_lines(text)
        # print(f'\x1b[4mPreprocessor\x1b[0m: Remove New Lines')

        text = self.remove_unicode_characters(text)

        text = self.remove_specific_patterns(text)
        # print(f'\x1b[4mPreprocessor\x1b[0m: Remove Specific Patterns')

        text = self.remove_nonalphanumeric_tokens(text)

        text = self.remove_multiple_whitespace(text)
        # print(f'\x1b[4mPreprocessor\x1b[0m: Remove Whitespaces')

        # text = self.remove_signatures(text)
        
        return text

    def remove_links(
        self,
        text: str,
        url_pattern: str = r'(?:http:|https:|www\.)[^\s"]+',
        html_url_pattern:str = r'(href|src|img)="([^"]+)"'
    ) -> str:
        """Remove links from text

        Args:
            text (str): text to remove links from
            links_pattern (str, optional): Pattern to find links in text. Defaults to r'(?:http:|https:|www\.)[^\s"]+'.
        
        Returns:
            text (str): text with links removed
        """
        
        if bool(re.search(html_url_pattern, text, flags = re.DOTALL | re.IGNORECASE)):
            text = re.sub(html_url_pattern, lambda match: f'{match.group(1)}="<URL>"', text, flags=re.DOTALL | re.IGNORECASE)
        
        return re.sub(url_pattern, '<URL>', text, flags = re.DOTALL | re.IGNORECASE)

    def replace_html_content(
        self,
        text: str,
        html_pattern: str = r'(?:<html>|<head>|<body>|<title>|<p>|<href|<font|<i>|<b>|<br>|<dl>)'
    ) -> str:
        """Find html tags if any in text, and replace with the text content in it

        Args:
            text (str): text to replace html content from
            html_pattern (str, optional): Pattern to find html tags in text. Defaults to r'(?:<html>|<head>|<body>|<title>|<p>|<href|<font|<i>|<b>|<br>|<dl>)'.
        
        Returns:
            text (str): text with replaced html contents
        """
        
        if bool(re.search(html_pattern, text, flags = re.DOTALL | re.IGNORECASE)):
            return html2text.html2text(text)
        
        return text

    def remove_new_lines(
        self,
        text: str,
    ) -> str:
        """Remove new lines from text

        Args:
            text (str): text to remove new lines from
        
        Returns:
            text (str): text with new lines removed
        """

        text = re.sub(r'\r\n', ' ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\r', ' ', text)
        return text

    def remove_unicode_characters(
        self,
        text: str
    ) -> str:
        """Remove unicode characters from text

        Args:
            text (str): text to remove unicode characters from
        
        Returns:
            text (str): text with unicode characters removed
        """

        return re.sub(r'[^\x00-\x7F]+', ' ', text)

    def remove_specific_patterns(
        self,
        text: str,
    ) -> str:
        """Remove specific patterns from text

        Args:
            text (str): text to remove patterns from
        
        Returns:
            text (str): text with patterns removed
        """

        message_type = [
            r'-+Original Message-+'
        ]

        header_type = [
            r'From:.+?(?=Sent:)',
            r'Sent:.+?(?=To:)',
            r'To:.+?(?=Cc:)',
            r'Cc:.+?(?=Subject:)',
            r'Subject:.+?(\n|$)'
        ]

        for pattern in message_type + header_type:
            text = re.sub(pattern, ' ', text, flags = re.DOTALL | re.IGNORECASE)

        return text

    def remove_nonalphanumeric_tokens(
        self,
        text: str
    ) -> str:
        """Remove non-alphanumeric tokens from text

        Args:
            text (str): text to remove non-alphanumeric tokens from
        
        Returns:
            text (str): text with non-alphanumeric tokens removed
        """

        return re.sub(r'\s[^\w\s]+\s', ' ', text, flags = re.DOTALL | re.IGNORECASE)

    def remove_multiple_whitespace(
        self,
        text: str,
    ) -> str:
        """Remove multiple whitespace from text

        Args:
            text (str): text to remove multiple whitespace from
        
        Returns:
            text (str): text with multiple whitespace removed
        """

        text = re.sub(r'\s+', ' ', text)
        return text

    # def remove_people_names(
    #     self,
    #     text: str,
    # ) -> str:

    # def remove_signatures(
    #     self,
    #     text: str, 
    #     threshold: float = .9
    # ) -> str:
    #     """Remove signatures from text
        
    #     Args:
    #         text (str): text to remove signatures from
    #         threshold (float | int, optional): Threshold to determine if a sentence is a signature block. Defaults to .9.

    #     Returns:
    #         text (str): text with signatures removed
    #     """
        
    #     output_text = ''
    #     pos_tagger = English()  # part-of-speech tagger
    #     sentences = sent_tokenize(text)  # convert to sentences
        
    #     tagger = spacy.load('en_core_web_sm')

    #     for sentence in sentences:
    #         if self.get_prob_block(sentence, tagger) < threshold:
    #             output_text += sentence + '. '
        
    #     return output_text

    # def get_prob_block(
    #     self,
    #     text: str, 
    #     pos_tagger: spacy.load('en_core_web_sm')
    # ) -> float:
    #     """Calculate probability that a sentence is an email block.
        
    #     https://spacy.io/usage/linguistic-features

    #     Args:
    #         text (str): text to calculate probability for
    #         pos_tagger (spacy.load('en_core_web_sm')): part-of-speech tagger

    #     Returns:
    #         (float): probability(signature block | line)
    #     """
        
    #     doc = pos_tagger(text)
    #     verb_count = np.sum([token.pos_ != "VERB" for token in doc])
    #     return float(verb_count) / len(doc)
    
