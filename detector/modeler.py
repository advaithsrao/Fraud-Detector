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

from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertModel
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader, TensorDataset#, SubsetRandomSampler
import torch.nn.functional as F

import wandb
from mlflow.sklearn import save_model
from scipy.sparse import hstack

from utils.util_modeler import Word2VecEmbedder, TPSampler


class NNFraudModel:
    def __init__(
        self, 
        num_labels: int = 2,
        path: str = '',
        model_name='roberta-base', 
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
        
        if self.model_name == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        elif self.model_name == 'distilbert-base-uncased':
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        self.device = torch.device(self.device)

        self.model = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

        self.model.to(device)
        
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

        # Initialize the optimizer and learning rate scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=self.epsilon)
        total_steps = len(train_dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Initialize variables for early stopping
        best_validation_loss = float("inf")
        patience = 5  # Number of epochs to wait for improvement
        wait = 0

        for epoch in range(self.num_epochs):
            print(f'{"="*20} Epoch {epoch + 1}/{self.num_epochs} {"="*20}')

            # Training loop
            self.model.train()
            total_train_loss = 0

            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(self.device, dtype=torch.float)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device, dtype=torch.float)

                # Forward pass
                outputs = self.model(b_input_ids)
                
                logits = F.sigmoid(outputs)   # Apply sigmoid to the final output

                # Compute binary cross-entropy loss
                loss = F.binary_cross_entropy(logits, b_labels)

                total_train_loss += loss.item()

                # Backward pass
                loss.backward()

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

            # Evaluation loop
            self.model.eval()
            total_eval_accuracy = 0
            total_eval_loss = 0

            for batch in validation_dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                with torch.no_grad():
                    outputs = self.model(b_input_ids)
                    # Apply sigmoid to the final output
                    logits = F.sigmoid(outputs)   
                    # Compute binary cross-entropy loss
                    loss = F.binary_cross_entropy(logits, b_labels.float())
                
                total_eval_loss += loss.item()
                
                # logits = logits.detach().cpu().numpy()
                # label_ids = b_labels.detach().cpu().numpy()

                total_eval_accuracy += self.accuracy(logits, b_labels)

            if len(validation_dataloader) > 0:
                avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
                print(f'Validation Accuracy: {avg_val_accuracy:.4f}')

                avg_val_loss = total_eval_loss / len(validation_dataloader)
                print(f'Validation Loss: {avg_val_loss:.4f}')

                # Early stopping check
                if avg_val_loss < best_validation_loss:
                    best_validation_loss = avg_val_loss
                    wait = 0
                else:
                    wait += 1

                if wait >= patience:
                    print(f'Early stopping after {patience} epochs without improvement.')
                    break
            else:
                print('No validation data provided.')
                avg_val_accuracy = 0
                avg_val_loss = 0

            if wandb is not None:
                wandb.log({
                    'epoch': epoch, 
                    'train_loss': avg_train_loss, 
                    'val_loss': avg_val_loss,
                    'val_accuracy': avg_val_accuracy,
                })

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
                # Apply sigmoid to the final output
                logits = F.sigmoid(outputs)

            logits = logits.detach().cpu().numpy()

            # Apply a threshold (e.g., 0.5) to convert logits to class predictions
            class_predictions = np.argmax(logits, axis=1)
            
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
        
        model_state_dict_path = os.path.join(path, 'model_state_dict.pth')
        
        torch.save(self.model.state_dict(), model_state_dict_path)
    
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


class RobertaFraudModel:
    def __init__(
        self, 
        num_labels: int = 2,
        path: str = '',
        model_name='roberta-base', 
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
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)

        if self.path != '':
            self.model = RobertaModel.from_pretrained(self.path).to(self.device)
        else:
            self.model = RobertaModel.from_pretrained(self.model_name).to(self.device)
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_labels)
        )

        self.classification_head.to(self.device)
        
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

        # Initialize the optimizer and learning rate scheduler
        optimizer = AdamW(list(self.model.parameters()) + list(self.classification_head.parameters()),
                          lr=self.learning_rate, eps=self.epsilon)
        total_steps = len(train_dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Initialize variables for early stopping
        best_validation_loss = float("inf")
        patience = 5  # Number of epochs to wait for improvement
        wait = 0

        for epoch in range(self.num_epochs):
            print(f'{"="*20} Epoch {epoch + 1}/{self.num_epochs} {"="*20}')

            # Training loop
            self.model.train()
            self.classification_head.train()
            total_train_loss = 0

            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Forward pass
                outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                last_hidden_states = outputs.last_hidden_state

                # Apply classification head
                logits = self.classification_head(last_hidden_states[:, 0, :])

                loss = F.cross_entropy(logits, b_labels)

                total_train_loss += loss.item()

                # Backward pass
                loss.backward()

                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.classification_head.parameters()), 1.0)

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

            # Evaluation loop
            self.model.eval()
            self.classification_head.eval()
            total_eval_accuracy = 0
            total_eval_loss = 0

            for batch in validation_dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                with torch.no_grad():
                    outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                    last_hidden_states = outputs.last_hidden_state

                    # Apply classification head
                    logits = self.classification_head(last_hidden_states[:, 0, :])

                    loss = F.cross_entropy(logits, b_labels)

                    total_eval_loss += loss.item()
                    total_eval_accuracy += self.accuracy(logits, b_labels)

                total_eval_accuracy += self.accuracy(logits, b_labels)

            if len(validation_dataloader) > 0:
                avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
                print(f'Validation Accuracy: {avg_val_accuracy:.4f}')

                avg_val_loss = total_eval_loss / len(validation_dataloader)
                print(f'Validation Loss: {avg_val_loss:.4f}')

                # Early stopping check
                if avg_val_loss < best_validation_loss:
                    best_validation_loss = avg_val_loss
                    wait = 0
                else:
                    wait += 1

                if wait >= patience:
                    print(f'Early stopping after {patience} epochs without improvement.')
                    break
            else:
                print('No validation data provided.')
                avg_val_accuracy = 0
                avg_val_loss = 0

            if wandb is not None:
                wandb.log({
                    'epoch': epoch, 
                    'train_loss': avg_train_loss, 
                    'val_loss': avg_val_loss,
                    'val_accuracy': avg_val_accuracy,
                })

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
                last_hidden_states = outputs.last_hidden_state

                # Apply classification head
                logits = self.classification_head(last_hidden_states[:, 0, :])

            logits = logits.detach().cpu().numpy()

            # Apply a threshold (e.g., 0.5) to convert logits to class predictions
            class_predictions = np.argmax(logits, axis=1)
            
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


class DistilbertFraudModel:
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
            self.model = DistilBertModel.from_pretrained(self.path).to(self.device)
        else:
            self.model = DistilBertModel.from_pretrained(self.model_name).to(self.device)

        self.classification_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_labels)
        )

        self.classification_head.to(self.device)
        
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

        # Initialize the optimizer and learning rate scheduler
        optimizer = AdamW(list(self.model.parameters()) + list(self.classification_head.parameters()),
                          lr=self.learning_rate, eps=self.epsilon)
        total_steps = len(train_dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Initialize variables for early stopping
        best_validation_loss = float("inf")
        patience = 5  # Number of epochs to wait for improvement
        wait = 0

        for epoch in range(self.num_epochs):
            print(f'{"="*20} Epoch {epoch + 1}/{self.num_epochs} {"="*20}')

            # Training loop
            self.model.train()
            self.classification_head.train()
            total_train_loss = 0

            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Forward pass
                outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                last_hidden_states = outputs.last_hidden_state

                # Apply classification head
                logits = self.classification_head(last_hidden_states[:, 0, :])

                loss = F.cross_entropy(logits, b_labels)

                total_train_loss += loss.item()

                # Backward pass
                loss.backward()

                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.classification_head.parameters()), 1.0)

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

            # Evaluation loop
            self.model.eval()
            self.classification_head.eval()
            total_eval_accuracy = 0
            total_eval_loss = 0

            for batch in validation_dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                with torch.no_grad():
                    outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                    last_hidden_states = outputs.last_hidden_state

                    # Apply classification head
                    logits = self.classification_head(last_hidden_states[:, 0, :])

                    loss = F.cross_entropy(logits, b_labels)

                    total_eval_loss += loss.item()
                    total_eval_accuracy += self.accuracy(logits, b_labels)

                total_eval_accuracy += self.accuracy(logits, b_labels)

            if len(validation_dataloader) > 0:
                avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
                print(f'Validation Accuracy: {avg_val_accuracy:.4f}')

                avg_val_loss = total_eval_loss / len(validation_dataloader)
                print(f'Validation Loss: {avg_val_loss:.4f}')

                # Early stopping check
                if avg_val_loss < best_validation_loss:
                    best_validation_loss = avg_val_loss
                    wait = 0
                else:
                    wait += 1

                if wait >= patience:
                    print(f'Early stopping after {patience} epochs without improvement.')
                    break
            else:
                print('No validation data provided.')
                avg_val_accuracy = 0
                avg_val_loss = 0

            if wandb is not None:
                wandb.log({
                    'epoch': epoch, 
                    'train_loss': avg_train_loss, 
                    'val_loss': avg_val_loss,
                    'val_accuracy': avg_val_accuracy,
                })

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
                last_hidden_states = outputs.last_hidden_state

                # Apply classification head
                logits = self.classification_head(last_hidden_states[:, 0, :])

            logits = logits.detach().cpu().numpy()

            # Apply a threshold (e.g., 0.5) to convert logits to class predictions
            class_predictions = np.argmax(logits, axis=1)
            
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


class SVMFraudModel:
    def __init__(
        self,
        num_labels: int = 2,
        kernel='linear',
        C=1.0,
    ):
        self.num_labels = num_labels
        self.kernel = kernel
        self.C = C
        self.vectorizer = Word2VecEmbedder()
        self.model = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', SVC(kernel=self.kernel, C=self.C, probability=True, random_state=42, verbose=True, class_weight='balanced'))
        ])

    def train(
        self,
        body: pd.Series | list[str],
        label: pd.Series | list[int],
    ):
        """Trains the SVM model.

        Args:
            body (pd.Series | list[str]): The body of the email.
            label (pd.Series | list[int]): The label of the email.

        Raises:
            ValueError: If the body and label are not of the same size.
        """
        if isinstance(body, pd.Series):
            body = body.tolist()
        if isinstance(label, pd.Series):
            label = label.tolist()

        # Train the SVM model
        self.model.fit(body, label)

    def predict(
        self,
        body: pd.Series | list[str],
    ):
        """Predicts the labels of the given data.

        Args:
            body (pd.Series | list[str]): The body of the email.

        Returns:
            np.array: The predictions of the model.
        """
        if isinstance(body, pd.Series):
            body = body.tolist()

        # Make predictions using the trained SVM model
        predictions = self.model.predict(body)

        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()

        return predictions

    def save_model(
        self,
        path: str,
    ):
        """Saves the model to the given path.

        Args:
            path (str): The path to save the model to.
        """

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        save_model(self.model, path)
