#usage: python3 -m pipelines.svm_trainer --num_labels 2 --C 10 --kernel 'rbf' --save_path '/tmp/model'
import sys
sys.path.append('..')

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pandas as pd
import sys
import os

from detector.data_loader import LoadEnronData, LoadPhishingData, LoadSocEnggData
from detector.labeler import EnronLabeler
from detector.modeler import SVMModel
from detector.preprocessor import Preprocessor
from utils.util_modeler import evaluate_and_log, get_f1_score

import wandb
import argparse
import configparser
config = configparser.ConfigParser()
config.read(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../config.ini'
    )
)

def parse_args():
    parser = argparse.ArgumentParser(description="SVM Model Fraud Detector Pipeline")
    parser.add_argument("--save_path", "-s", type=str, default='/tmp/', help="Output save path")
    parser.add_argument("--num_labels", "-l", type=int, default=2, help="Number of labels")
    parser.add_argument("--C", "-C", type=int, default=1, help="Regularization parameter")
    parser.add_argument("--kernel", "-k", type=str, default='rbf', help="Kernel to use in the algorithm ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')")
    return parser.parse_args()

def load_data():
    if os.path.exists(
        os.path.join(
            os.path.dirname(__file__),
            '../data/fraud_detector_data.csv'
        )
    ):
        data = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__),
                '../data/fraud_detector_data.csv'
            )
        )
    else:
        data = {
            loader.__name__: loader().__call__() for loader in [LoadEnronData, LoadPhishingData, LoadSocEnggData]
        }
    return data

def label_and_preprocess_data(data):
    if not os.path.exists(
        os.path.join(
            os.path.dirname(__file__),
            '../data/fraud_detector_data.csv'
        )
    ):
        # Run Enron Labeler
        data['LoadEnronData'] = EnronLabeler(data['LoadEnronData'], needs_preprocessing=True)()

        # Preprocess the other 2 datasets
        data['LoadPhishingData']['Body'] = data['LoadPhishingData']['Body'].swifter.apply(Preprocessor())
        data['LoadSocEnggData']['Body'] = data['LoadSocEnggData']['Body'].swifter.apply(Preprocessor())

        # Concatenate the 3 data sources into 1
        data = pd.concat(
            [
                df for df in data.values()
            ],
            axis=0,
            ignore_index=True
        )
    return data

def data_split(data):
    if not os.path.exists(
        os.path.join(
            os.path.dirname(__file__),
            '../data/fraud_detector_data.csv'
        )
    ):
        # For gold_fraud_set, take first 500 emails from Phishing Data and 500 emails from Social Engineering Data
        gold_fraud = pd.concat(
            [
                data[data['Source'] == 'Phishing Data'][data['Label'] == 1].head(500),
                data[data['Source'] == 'Social Engineering Data'][data['Label'] == 1].head(500)
            ],
            axis=0,
            ignore_index=True
        )
        gold_fraud['Split'] = 'Gold Fraud'

        # For sanity_set, take first 5000 emails with Sender-Type = 'Internal'
        sanity = data[
            (data['Sender-Type'] == 'Internal') & (data['Source'] == 'Enron Data')
        ].head(5000)
        sanity['Split'] = 'Sanity'

        # For train_set, take all data not in gold_fraud_set and sanity_set
        train = data[
            ~data['Mail-ID'].isin(gold_fraud['Mail-ID']) & ~data['Mail-ID'].isin(sanity['Mail-ID'])
        ]

        train['Split'] = 'Train'

        #drop train examples with Label=1 and Body less than 4 words
        train = train[~((train['Label'] == 1) & (train['Body'].str.split().str.len() < 4))]

        train = train.reset_index(drop=True)

    else:
        train = data[data['Split'] == 'Train']
        gold_fraud = data[data['Split'] == 'Gold Fraud']
        sanity = data[data['Split'] == 'Sanity']
        
    return train, sanity, gold_fraud

def train_model(train_data, hyper_params):
    run = wandb.init(config=hyper_params)
    model = SVMModel(**hyper_params)

    # Call your code that produces output
    model.train(body=train_data['Body'], label=train_data['Label'])
    return model

def test_and_save_model(train_data, sanity_data, gold_fraud_data, save_path):
    # Define a dictionary to store the f1 scores
    f1_scores = {}
    
    # Define a dictionary to store the predictions, true labels for each dataset
    true_pred_map = {
        'train':{},
        'sanity':{},
        'gold_fraud':{}
    }

    os.makedirs(os.path.join(save_path,'logs'), exist_ok=True)

    # Save the model and logs to the date folder
    model.save_model(os.path.join(save_path,'model'))

    true_pred_map['train']['true'] = train_data['Label'].tolist()
    true_pred_map['train']['pred'] = model.predict(body=train_data['Body'])

    evaluate_and_log(x=train_data['Body'].tolist(), y_true=true_pred_map['train']['true'], y_pred=true_pred_map['train']['pred'], filename=os.path.join(save_path,'logs/train.log'), experiment=run)
    f1_scores['train'] = get_f1_score(y_true=true_pred_map['train']['true'], y_pred=true_pred_map['train']['pred'])

    true_pred_map['sanity']['true'] = sanity_data['Label'].tolist()
    true_pred_map['sanity']['pred'] = model.predict(body=sanity_data['Body'])
    evaluate_and_log(x=sanity_data['Body'].tolist(), y_true=true_pred_map['sanity']['true'], y_pred=true_pred_map['sanity']['pred'], filename=os.path.join(save_path,'logs/sanity.log'), experiment=run)
    f1_scores['sanity'] = get_f1_score(y_true=true_pred_map['sanity']['true'], y_pred=true_pred_map['sanity']['pred'])

    true_pred_map['gold_fraud']['true'] = gold_fraud_data['Label'].tolist()
    true_pred_map['gold_fraud']['pred'] = model.predict(body=gold_fraud_data['Body'])
    evaluate_and_log(x=gold_fraud_data['Body'].tolist(), y_true=true_pred_map['gold_fraud']['true'], y_pred=true_pred_map['gold_fraud']['pred'], filename=os.path.join(save_path,'logs/gold_fraud.log'), experiment=run)
    f1_scores['gold_fraud'] = get_f1_score(y_true=true_pred_map['gold_fraud']['true'], y_pred=true_pred_map['gold_fraud']['pred'])

    return f1_scores, true_pred_map

def dump_logs_to_wandb(hyper_params, f1_scores, true_pred_map, save_path):
    # Log the hyperparameters and f1 scores to Weights and Biases
    all_params = {**hyper_params, **f1_scores}
    run.config.update(all_params)

    # Log the model to Weights and Biases
    model_path = os.path.join(save_path, 'model')
    model_artifact = wandb.Artifact("fraud-detector-model", type="model")
    model_artifact.add_dir(model_path)
    run.use_artifact(model_artifact)

    # Log the log files to Weights and Biases
    logs_path = os.path.join(save_path,'logs')
    log_artifact = wandb.Artifact("fraud-detector-logs", type="logs")
    log_artifact.add_dir(logs_path)
    run.use_artifact(log_artifact)

    # Log confusion matrices
    # run.log({
    #     "train_confusion_matrix": wandb.plot.confusion_matrix(true_pred_map['train']['true'], true_pred_map['train']['pred']),
    #     "sanity_confusion_matrix": wandb.plot.confusion_matrix(true_pred_map['sanity']['true'], true_pred_map['sanity']['pred']),
    #     "gold_fraud_confusion_matrix": wandb.plot.confusion_matrix(true_pred_map['gold_fraud']['true'], true_pred_map['gold_fraud']['pred'])
    # })

if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()
    
    # Define model hyperparameters
    hyper_params = {
        'num_labels': args.num_labels,
        'C': args.C,
        'kernel': args.kernel
    }

    # Log in to Weights and Biases
    wandbdict = {
        'key': os.getenv('WANDB_API_KEY'),
        'entity': os.getenv('WANDB_ENTITY'),
        'project': os.getenv('WANDB_PROJECT'),
    }
    wandb.login(key=wandbdict['key'])
    run = wandb.init(project=wandbdict['project'], entity=wandbdict['entity'])

    # Define a variable to store the trained model
    model = None

    # Get the current date
    date = datetime.now().strftime("%Y-%m-%d")

    # Create date folder in save path
    save_path = args.save_path
    save_path = os.path.join(save_path, f'{date}')
    os.makedirs(save_path, exist_ok=True)

    # Load the data
    data = load_data()

    # Label and preprocess the data
    data = label_and_preprocess_data(data)

    # Split the data into train, sanity, and gold_fraud sets
    train_data, sanity_data, gold_fraud_data = data_split(data)

    # Train the model
    model = train_model(train_data, hyper_params)

    # Test the model
    f1_scores, true_pred_map = test_and_save_model(train_data, sanity_data, gold_fraud_data, save_path)

    # Dump the logs to Weights and Biases
    dump_logs_to_wandb(hyper_params, f1_scores, true_pred_map, save_path)

    # Close the Weights and Biases run
    run.finish()

