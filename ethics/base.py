import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import shutil
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
import torch
from torch import nn

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertModel
from transformers import AdamW,get_linear_schedule_with_warmup

from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import wandb
from mlflow.sklearn import save_model
from scipy.sparse import hstack


class BaseDistilbertModel(nn.Module):
    def __init__(self, num_labels, model_name='distilbert-base-uncased', device = 'cuda'):
        super(BaseDistilbertModel, self).__init__()

        # Load pre-trained RobertaModel
        self.model = DistilBertModel.from_pretrained(model_name).to(device)

        for param in self.model.parameters():
            param.requires_grad = False

        # Define classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        # Get model outputs
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state

        # Apply classification head
        logits = self.classification_head(last_hidden_states[:, 0, :])

        return logits