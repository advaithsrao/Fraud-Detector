import sys
sys.path.append("..")

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from numpy import array
import pytest

from utils.util_modeler import evaluate_and_log, get_f1_score, get_classification_report_confusion_matrix, Word2VecEmbedder, TPSampler


@pytest.fixture
def x():
    return ['Give me your account number quick', 'Give me your account number quick']

@pytest.fixture
def y_true():
    return [1, 1]

@pytest.fixture
def y_pred():
    return [0, 1]

@pytest.fixture
def mail():
    return """
    I would like to get some of the timing issues resolved prior to
    implementing "No Tag, No Flow."  The problems seem to be isolated, but it
    only takes a single entity to create huge problems for everyone involved.
    
    Joe Smith | Strategy & Business Development
    111 Market St. Suite 111| San Francisco, CA 94103
    M: 111.111.1111| joe@foobar.com
    """

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

    assert (conf_matrix == np.array([[0, 0], [1, 1]])).all()

def test_evaluate_and_log(x, y_true, y_pred):
    evaluate_and_log(x, y_true, y_pred, '/tmp/test.log')
    assert os.path.exists('/tmp/test.log')

def test_word2vec_embedding(mail):
    embedder = Word2VecEmbedder()
    embedding = embedder.transform(mail)[0]
    assert len(embedding) == 300

def test_tp_sampler():
    sampler = TPSampler(class_labels=[0,1,0,1,0,1])
    assert sampler.__len__() == 3

if __name__ == "__main__":
    pytest.main()
