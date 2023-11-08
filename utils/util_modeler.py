import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import gensim.downloader
from nltk.tokenize import RegexpTokenizer
import wandb
# from torch.utils.data import Sampler
from sklearn.utils.class_weight import compute_sample_weight

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


class Word2VecEmbedder:
    def __init__(
        self,
        model_name: str = 'word2vec-google-news-300',
        tokenizer: RegexpTokenizer(r'\w+') = RegexpTokenizer(r'\w+')
    ):
        self.model = gensim.downloader.load(model_name)
        self.tokenizer = tokenizer

    def fit_transform(
        self, 
        text: str,
        
    ):
        """Calculate Word2Vec embeddings for the given text.

        Args:
            text (str): text document.

        Returns:
            np.ndarray: Word2Vec embeddings for the input text.
        """

        embedding = []

        # Initialize an array to store Word2Vec embeddings for the input text
        words = self.tokenizer.tokenize(text)  # Tokenize the document
        word_vectors = [self.model[word] if word in self.model else np.zeros(self.model.vector_size) for word in words]
        document_embedding = np.mean(word_vectors, axis=0)  # Calculate the mean of word embeddings for the document
        embedding.append(document_embedding)

        return np.array(embedding)


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

    def __iter__(
        self
    ):
        """Iterate through the sampled indices.

        Returns:
            iter: The indices of the sampled data.
        """
        
        num_samples = len(self.tp_indices)
        tp_batch_size = int(self.tp_ratio * self.batch_size)
        non_tp_batch_size = self.batch_size - tp_batch_size
        sampled_indices = []

        while len(sampled_indices) < num_samples:
            tp_indices = np.random.choice(self.tp_indices, tp_batch_size, replace=True)
            non_tp_indices = np.random.choice(self.non_tp_indices, non_tp_batch_size, replace=True)
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
