import sys
sys.path.append("..")

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from numpy import array
import pytest

from utils.util_modeler import evaluate_and_log, get_f1_score, get_classification_report_confusion_matrix


@pytest.fixture
def x():
    return ['Give me your account number quick', 'Give me your account number quick']

@pytest.fixture
def y_true():
    return [1, 1]

@pytest.fixture
def y_pred():
    return [0, 1]

def test_get_f1_score(y_true, y_pred):
    f1_score = get_f1_score(y_true, y_pred)
    assert round(f1_score,3) == 0.667

def test_get_classification_report_confusion_matrix(y_true, y_pred):
    class_report, conf_matrix = get_classification_report_confusion_matrix(y_true, y_pred)

    assert class_report == {
        '0': {
            'precision': 0.0, 
            'recall': 0.0, 
            'f1-score': 0.0, 
            'support': 0.0
        }, 
        '1': {
            'precision': 1.0, 
            'recall': 0.5, 
            'f1-score': 0.6666666666666666, 
            'support': 2.0
        }, 
        'accuracy': 0.5, 
        'macro avg': {
            'precision': 0.5, 
            'recall': 0.25, 
            'f1-score': 0.3333333333333333, 
            'support': 2.0
        }, 
        'weighted avg': {
            'precision': 1.0, 
            'recall': 0.5, 
            'f1-score': 0.6666666666666666, 
            'support': 2.0
        }
    }

    assert conf_matrix.shape() == (2,2)
    assert conf_matrix[0][0] == 0
    assert conf_matrix[0][1] == 0
    assert conf_matrix[1][0] == 1
    assert conf_matrix[1][1] == 1

def test_evaluate_and_log(x, y_true, y_pred):
    evaluate_and_log(x, y_true, y_pred, '/tmp/test.log')
    assert os.path.exists('/tmp/test.log')

if __name__ == "__main__":
    pytest.main()
