import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

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
    filename: str
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

    for i in mismatched_indices:
        # Format the mismatched example in a code block
        mismatched_example = f"\nActual: {y_true[i]}\nPredicted: {y_pred[i]}\n\nText: {x[i]}\n\n"
        mismatched_examples.append(mismatched_example)

    # Format the results for logging
    log_content = f"---------Classification Report---------\n{classification_report(y_true, y_pred)}\n\n"
    log_content += f"---------Confusion Matrix---------\n{conf_matrix}\n\n"
    log_content += "---------Mismatched Examples---------\n\n"
    log_content += "\n\n".join(mismatched_examples)

    # Save the results to the log file
    with open(filename, 'w') as log_file:
        log_file.write(log_content)