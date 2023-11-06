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
from detector.modeler import RobertaModel
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
    parser = argparse.ArgumentParser(description="Fraud Detector Pipeline")
    parser.add_argument("--num_labels", "-l", type=int, default=2, help="Number of labels")
    parser.add_argument("--num_epochs", "-e", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--device", "-d", type=str, default='cpu', help="Device to train the model on: 'cpu', 'cuda' or 'gpu'")
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
    else:
        train = data[data['Split'] == 'Train']
        gold_fraud = data[data['Split'] == 'Gold Fraud']
        sanity = data[data['Split'] == 'Sanity']
    return train, sanity, gold_fraud

def train_model(train_data, hyper_params):
    run = wandb.init(config=hyper_params)
    model = RobertaModel(**hyper_params)

    # os.makedirs(f'/tmp/{date}/logs', exist_ok=True)

    # # Define a log file path
    # log_filename = f"/tmp/{date}/logs/model_training.log"

    # # Create or open the log file in write mode
    # log_file = open(log_filename, "w")

    # # Redirect stdout to the log file
    # sys.stdout = log_file

    # Call your code that produces output
    model.train(body=train_data['Body'], label=train_data['Label'], validation_size=0.2, wandb=run)

    # Restore the original stdout
    # sys.stdout = sys.__stdout__

    # Close the log file
    # log_file.close()
    return model

def test_model(train_data, sanity_data, gold_fraud_data):
    # Define a dictionary to store the f1 scores
    f1_scores = {}
    
    # Define a dictionary to store the predictions, true labels for each dataset
    true_pred_map = {
        'train':{},
        'sanity':{},
        'gold_fraud':{}
    }

    os.makedirs(f'/tmp/{date}/logs', exist_ok=True)

    # Save the model and logs to the date folder
    model.save_model(f'/tmp/{date}/model')

    true_pred_map['train']['true'] = train_data['Label'].tolist()
    true_pred_map['train']['pred'] = model.predict(body=train_data['Body'])

    evaluate_and_log(x=train_data['Body'].tolist(), y_true=true_pred_map['train']['true'], y_pred=true_pred_map['train']['pred'], filename=f'/tmp/{date}/logs/train.log')
    f1_scores['train'] = get_f1_score(y_true=true_pred_map['train']['true'], y_pred=true_pred_map['train']['pred'])

    true_pred_map['sanity']['true'] = sanity_data['Label'].tolist()
    true_pred_map['sanity']['pred'] = model.predict(body=sanity_data['Body'])
    evaluate_and_log(x=sanity_data['Body'].tolist(), y_true=true_pred_map['sanity']['true'], y_pred=true_pred_map['sanity']['pred'], filename=f'/tmp/{date}/logs/sanity.log')
    f1_scores['sanity'] = get_f1_score(y_true=true_pred_map['sanity']['true'], y_pred=true_pred_map['sanity']['pred'])

    true_pred_map['gold_fraud']['true'] = gold_fraud_data['Label'].tolist()
    true_pred_map['gold_fraud']['pred'] = model.predict(body=gold_fraud_data['Body'])
    evaluate_and_log(x=gold_fraud_data['Body'].tolist(), y_true=true_pred_map['gold_fraud']['true'], y_pred=true_pred_map['gold_fraud']['pred'], filename=f'/tmp/{date}/logs/gold_fraud.log')
    f1_scores['gold_fraud'] = get_f1_score(y_true=true_pred_map['gold_fraud']['true'], y_pred=true_pred_map['gold_fraud']['pred'])

    return f1_scores, true_pred_map

def dump_logs_to_wandb(hyper_params, f1_scores, true_pred_map):
    # Log the hyperparameters and f1 scores to Weights and Biases
    all_params = {**hyper_params, **f1_scores}
    run.config.update(all_params)

    # Log the model to Weights and Biases
    model_path = f'/tmp/{date}/model'
    model_artifact = wandb.Artifact("fraud-detector-model", type="model")
    model_artifact.add_dir(model_path)
    run.use_artifact(model_artifact)

    # Log the log files to Weights and Biases
    logs_path = f'/tmp/{date}/logs'
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
    device = args.device
    device = device if device != 'gpu' else 'cuda'
    
    # Define model hyperparameters
    hyper_params = {
        'num_labels': args.num_labels,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'device': args.device,
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

    # Create date folder in /tmp
    os.makedirs(f'/tmp/{date}', exist_ok=True)

    # Load the data
    data = load_data()

    # Label and preprocess the data
    data = label_and_preprocess_data(data)

    # Split the data into train, sanity, and gold_fraud sets
    train_data, sanity_data, gold_fraud_data = data_split(data)

    # Train the model
    model = train_model(train_data, hyper_params)

    # Test the model
    f1_scores, true_pred_map = test_model(train_data, sanity_data, gold_fraud_data)

    # Dump the logs to Weights and Biases
    dump_logs_to_wandb(hyper_params, f1_scores, true_pred_map)

    # Close the Weights and Biases run
    run.finish()
