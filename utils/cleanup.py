import re
from typing import Any


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
        text = self.remove_new_lines(text)
        text = self.remove_specific_patterns(text)
        text = self.remove_multiple_whitespace(text)
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
        