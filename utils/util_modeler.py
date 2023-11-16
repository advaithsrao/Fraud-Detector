import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import gensim.downloader
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string
import wandb
# from torch.utils.data import Sampler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('stopwords')

def get_f1_score(
    y_true: list[int],
    y_pred: list[int]
    ):
    """Returns the F1 score.

    Args:
        y_true (list[int]): The true labels.
        y_pred (list[int]): The predicted labels.

    Returns:
        float: The F1 score.
    """

    return f1_score(y_true, y_pred)

def get_classification_report_confusion_matrix(
    y_true: list[int],
    y_pred: list[int]
    ):
    """Returns the classification report and confusion matrix.

    Args:
        y_true (list[int]): The true labels.
        y_pred (list[int]): The predicted labels.

    Returns:
        tuple: The classification report and confusion matrix.
    """

    return classification_report(y_true, y_pred, output_dict=True), confusion_matrix(y_true, y_pred)

def evaluate_and_log(
    x: list[str], 
    y_true: list[int], 
    y_pred: list[int], 
    filename: str,
    experiment: wandb = None
    ):
    """Evaluates the model's performance and logs the results.

    Args:
        x (list[str]): The texts used for evaluation.
        y_true (list[int]): The actual labels.
        y_pred (list[int]): The predicted labels.
        filename (str): The name of the log file.
    """
    
    if len(x) != len(y_true) or len(x) != len(y_pred):
        raise ValueError("Input lists (x, y_true, and y_pred) must have the same length.")

    # Calculate the classification report and confusion matrix
    class_report, conf_matrix = get_classification_report_confusion_matrix(y_true, y_pred)

    # Find mismatched examples -> indices from y_pred and y_true where they are not the same
    mismatched_indices = np.where(np.array(y_true) != np.array(y_pred))[0]
    mismatched_examples = []
    
    if experiment is not None:
        table = wandb.Table(columns=["Actual", "Predicted", "Text"])

    for i in mismatched_indices:
        # Format the mismatched example in a code block
        mismatched_example = f"\nActual: {y_true[i]}\nPredicted: {y_pred[i]}\n\nText: {x[i]}\n\n"
        mismatched_examples.append(mismatched_example)

        if experiment is not None:
            table.add_data(y_true[i], y_pred[i], x[i])

    # Format the results for logging
    log_content = f"---------Classification Report---------\n{classification_report(y_true, y_pred)}\n\n"
    log_content += f"---------Confusion Matrix---------\n{conf_matrix}\n\n"
    log_content += "---------Mismatched Examples---------\n\n"
    log_content += "\n\n".join(mismatched_examples)

    # Log the table
    if experiment is not None:
        wandb.log({"Mismatched_Examples": table})

    # Save the results to the log file
    with open(filename, 'w') as log_file:
        log_file.write(log_content)

def calculate_document_embedding(doc, model, tokenizer, embed_size):
    """Calculates the document embedding for the given document.
    
    Utility function for below class - Word2VecEmbedder

    Args:
        doc (str): The document.
        model (gensim.models.keyedvectors.Word2VecKeyedVectors): The Word2Vec model.
        tokenizer (nltk.tokenize.regexp.RegexpTokenizer): The tokenizer.
        embed_size (int): The embedding size.

    Returns:
        np.ndarray: The document embedding.
    """
    
    doc_embed = np.zeros(embed_size)
    words = tokenizer.tokenize(doc)
    stopset = stopwords.words('english') + list(string.punctuation)

    #we lowercase the words specifically for OOV embeddings to be same for same words different case
    words = [word.lower() for word in words]
    words = [word for word in words if word not in stopset]

    word_count = 0
    for word in words:
        if word in model:
            doc_embed += model[word]
            word_count += 1
        
    return doc_embed / word_count if word_count != 0 else doc_embed


class Word2VecEmbedder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name: str = 'word2vec-google-news-300',
        tokenizer=RegexpTokenizer(r'\w+')
    ):
        self.model = gensim.downloader.load(model_name)
        self.tokenizer = tokenizer
        self.embed_size = 300

    def fit(
        self, 
        X, 
        y=None
    ):
        return self

    def transform(
        self, 
        X
    ):
        """Calculate Word2Vec embeddings for the given text.

        Args:
            X (list): List of text documents.

        Returns:
            np.ndarray: Word2Vec embeddings for the input text.
        """

        if isinstance(X, str):
            X = [X]

        return np.vstack([calculate_document_embedding(doc, self.model, self.tokenizer, self.embed_size) for doc in X])


class TPSampler:
    def __init__(
        self, 
        class_labels, 
        tp_ratio=0.1, 
        batch_size=32
    ):
        """A custom sampler to sample the training data.
        
        Args:
            class_labels (list[int]): The class labels of the training data.
            tp_ratio (float, optional): The ratio of true positives to sample. Defaults to 0.1.
            batch_size (int, optional): The batch size. Defaults to 32.

        Returns:
            iter: The indices of the sampled data.
        """

        self.tp_indices = [i for i, label in enumerate(class_labels) if label == 1]
        self.non_tp_indices = [i for i, label in enumerate(class_labels) if label == 0]
        self.tp_ratio = tp_ratio
        self.batch_size = batch_size

    def __iter__(self):
        """Iterate through the sampled indices.

        Returns:
            iter: The indices of the sampled data.
        """
        
        num_samples = len(self.tp_indices)
        tp_batch_size = int(self.tp_ratio * self.batch_size)
        non_tp_batch_size = self.batch_size - tp_batch_size
        sampled_indices = []

        while len(sampled_indices) < num_samples:
            tp_indices = np.random.choice(self.tp_indices, tp_batch_size, replace=False)
            non_tp_indices = np.random.choice(self.non_tp_indices, non_tp_batch_size, replace=False)
            batch_indices = np.concatenate((tp_indices, non_tp_indices))
            np.random.shuffle(batch_indices)
            sampled_indices.extend(batch_indices)

        return iter(sampled_indices)

    def __len__(
        self
    ):
        """Returns the total number of samples for the dataloader.

        Returns:
            int: The total number of samples for the dataloader.
        """
        
        return len(self.tp_indices)  # This defines the total number of samples for the dataloader
