import re

def remove_new_lines(
    text: str,
):
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
    