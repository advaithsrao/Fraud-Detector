import re
import html2text
from typing import Any

def add_subject_to_body(
    subject: str,
    body: str
) -> str:
    if subject in body:
        return body
    else:
        return subject + ' ' + body
    

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
        
        return text

    def remove_links(
        self,
        text: str
    ) -> str:
        """Remove links from text

        Args:
            text (str): text to remove links from
        
        Returns:
            text (str): text with links removed
        """

        return re.sub('(?:http\:|https\:|www\.)\S+', '<URL>', text, flags = re.DOTALL | re.IGNORECASE)

    def replace_html_content(
        self,
        text: str,
        # html_pattern: str = r'(?:<html>|<head>|<body>|<title>|<p>|<href'
    ) -> str:
        """Find html tags if any in text, and replace with the text content in it

        Args:
            text (str): text to replace html content from
        
        Returns:
            text (str): text with replaced html contents
        """
        
        return html2text.html2text(text)

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

        return re.sub(r'[^\x00-\x7F]+', '', text)

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

        return re.sub(r'\s[^\w\s]+\s', '', text, flags = re.DOTALL | re.IGNORECASE)

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
        
    
