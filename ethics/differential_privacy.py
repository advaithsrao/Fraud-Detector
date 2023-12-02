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
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import wandb
from mlflow.sklearn import save_model
from scipy.sparse import hstack

from utils.util_modeler import Word2VecEmbedder, TPSampler

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


class DistilbertPrivacyModel:
    def __init__(
        self, 
        num_labels=2, 
        path='', 
        model_name='distilbert-base-uncased', 
        learning_rate=2e-5, 
        epsilon=1e-8, 
        num_epochs=40, 
        batch_size=128, 
        device=None
    ):
        self.num_labels = num_labels
        self.path = path
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

        if not self.device and torch.cuda.is_available():
            self.device = 'cuda'
        elif not self.device:
            self.device = 'cpu'

        self.device = torch.device(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        if self.path != '':
            self.model = BaseModel.to(self.device)
        else:
            self.model = BaseModel.from_pretrained(self.model_name).to(self.device)

        self.privacy_engine = PrivacyEngine()
        
    def train(
        self, 
        body: pd.Series | list[str], 
        label: pd.Series | list[int], 
        validation_size=0.2,
        wandb=None
    ):
        """Trains the model using the given data.

        Args:
            body (pd.Series | list[str]): The body of the email.
            label (pd.Series | list[int]): The label of the email.
            validation_size (float, optional): The size of the validation set. Defaults to 0.2.
            wandb (wandb, optional): The wandb object. Defaults to None. If given, logs the training process to wandb.

        Raises:
            ValueError: If the body and label are not of the same size.
        """

        if isinstance(body, pd.Series):
            body = body.tolist()
        if isinstance(label, pd.Series):
            label = label.tolist()

        # Tokenize input texts and convert labels to tensors
        input_ids = []
        attention_masks = []
        label_ids = []

        for _body, _label in zip(body, label):
            # Tokenize the input text using the Roberta tokenizer
            inputs = self.tokenizer.encode_plus(
                _body,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            input_ids.append(inputs['input_ids'])
            attention_masks.append(inputs['attention_mask'])
            label_ids.append(torch.tensor(_label))  # Convert the label to a tensor

        # Convert lists to tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        label_ids = torch.stack(label_ids)

        # Split the data into train and validation sets
        dataset = TensorDataset(input_ids, attention_masks, label_ids)
        dataset_size = len(dataset)
        val_size = int(validation_size * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create data loaders for training and validation data
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Initialize the Privacy engine, optimizer and learning rate scheduler
        optimizer = AdamW(list(self.model.parameters()),
                          lr=self.learning_rate, eps=self.epsilon)
        
        total_steps = len(train_dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        MAX_GRAD_NORM = 0.1

        self.model, optimizer, train_dataloader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            target_delta=1/len(train_dataloader),
            target_epsilon=self.epsilon, 
            epochs=self.num_epochs,
            max_grad_norm=MAX_GRAD_NORM,
        )

        # Initialize variables for early stopping
        best_validation_loss = float("inf")
        patience = 5  # Number of epochs to wait for improvement
        wait = 0

        for epoch in range(self.num_epochs):
            print(f'{"="*20} Epoch {epoch + 1}/{self.num_epochs} {"="*20}')

            # Training loop
            self.model.train()
            total_train_loss = 0

            with BatchMemoryManager(
                data_loader=train_dataloader, 
                max_physical_batch_size=self.batch_size, 
                optimizer=optimizer
            ) as memory_safe_data_loader:
                for step, batch in enumerate(memory_safe_data_loader):
                    optimizer.zero_grad()
                    b_input_ids = batch[0].to(self.device, dtype=torch.long)
                    b_input_mask = batch[1].to(self.device, dtype=torch.long)
                    b_labels = batch[2].to(self.device, dtype=torch.long)

                    # Forward pass
                    outputs = self.model(b_input_ids, attention_mask=b_input_mask)

                    loss = F.cross_entropy(outputs, b_labels)

                    total_train_loss += loss.item()

                    # Backward pass
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), 1.0)

                    # Update the model parameters
                    optimizer.step()

                    # Update the learning rate
                    scheduler.step()

                    if step % 100 == 0 and step != 0:
                        avg_train_loss = total_train_loss / 100
                        print(f'Step {step}/{len(train_dataloader)} - Average training loss: {avg_train_loss:.4f}')

                        total_train_loss = 0

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f'Training loss: {avg_train_loss:.4f}')

            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta=1e-5)
            print(f"For Model, ε = {epsilon:.2f}, δ = 1e-5 for α = {best_alpha}")

            # Evaluation loop
            self.model.eval()
            total_eval_accuracy = 0
            total_eval_loss = 0

            for batch in validation_dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                with torch.no_grad():
                    outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                    
                    big_val, big_idx = torch.max(outputs.data, dim=1)
                    
                    loss = F.cross_entropy(outputs, b_labels)

                    total_eval_loss += loss.item()

                total_eval_accuracy += self.accuracy(big_idx, b_labels)

            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print(f'Validation Accuracy: {avg_val_accuracy:.4f}')

            avg_val_loss = total_eval_loss / len(validation_dataloader)
            print(f'Validation Loss: {avg_val_loss:.4f}')

            if wandb is not None:
                wandb.log({
                    'epoch': epoch, 
                    'train_loss': avg_train_loss, 
                    'val_loss': avg_val_loss,
                    'val_accuracy': avg_val_accuracy,
                })

            # Early stopping check
            if avg_val_loss < best_validation_loss:
                best_validation_loss = avg_val_loss
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                print(f'Early stopping after {patience} epochs without improvement.')
                break

    def predict(
        self, 
        body: pd.Series | list[str]
    ):
        """Predicts the labels of the given data.

        Args:
            body (pd.Series | list[str]): The body of the email.

        Returns:
            np.array: The predictions of the model.
        """

        # If input_texts is a Pandas Series, convert it to a list
        if isinstance(body, pd.Series):
            body = body.tolist()

        input_ids = []
        attention_masks = []

        for _body in body:
            inputs = self.tokenizer.encode_plus(
                _body,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            input_ids.append(inputs['input_ids'])
            attention_masks.append(inputs['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.eval()
        predictions = []

        for batch in dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)

            with torch.no_grad():
                outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                big_val, big_idx = torch.max(outputs.data, dim=1)

            class_predictions = big_idx.detach().cpu().numpy()

            # Apply a threshold (e.g., 0.5) to convert logits to class predictions
            # class_predictions = np.argmax(logits, axis=1)
            
            predictions.extend(class_predictions.tolist())

        return predictions
    
    def save_model(
            self,
            path: str
    ):
        """Saves the model to the given path.

        Args:
            path (str): The path to save the model to.
        """

        # Check if the directory exists, and if not, create it
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # Save the transformer model and the classification head
        self.model.save_pretrained(path)
        torch.save(self.classification_head.state_dict(), os.path.join(path, 'classification_head.pth'))
    
    def accuracy(
        self, 
        preds, 
        labels
    ):
        """Calculates the accuracy of the model.

        Args:
            preds (torch.Tensor|numpy.ndarray): The predictions of the model.
            labels (torch.Tensor|numpy.ndarray): The labels of the data.

        Returns:
            float: The accuracy of the model.
        """

        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        
        _, preds = torch.max(preds, dim=1)
        
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))