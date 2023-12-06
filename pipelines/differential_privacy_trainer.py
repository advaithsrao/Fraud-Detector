#usage: python3 -m pipelines.distilbert_trainer --num_epochs 20 --batch_size 8 --n_estimators 100 --criterion gini --num_labels 2 --device 'cuda' --save_path '/tmp' --model_name 'distilbert' --use_aug 'True'
import sys
sys.path.append('..')

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pandas as pd
import sys
import os


from detector.data_loader import LoadEnronData, LoadPhishingData, LoadSocEnggData
from detector.labeler import EnronLabeler, MismatchLabeler
from ethics.differential_privacy import DistilbertPrivacyModel, RandomForestPrivacyModel
from detector.preprocessor import Preprocessor
from utils.util_modeler import evaluate_and_log, get_f1_score, Augmentor

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
    parser = argparse.ArgumentParser(description="Distilbert Model Fraud Detector Pipeline")
    parser.add_argument("--save_path", "-s", type=str, default='/tmp/', help="Output save path")
    parser.add_argument("--num_labels", "-l", type=int, default=2, help="Number of labels")
    parser.add_argument("--model_name", "-m", type=str, default='random_forest', help="Model Name. Options: ['distilbert','random_forest']")
    parser.add_argument("--num_epochs", "-e", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--n_estimators", "-n", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--criterion", "-c", type=str, default='gini', help="Function to measure the quality of a split")
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-05, help="Learning rate for the model")
    parser.add_argument("--device", "-d", type=str, default='cpu', help="Device to train the model on: 'cpu', 'cuda' or 'gpu'")
    parser.add_argument("--use_aug", "-u", type=bool, default=False, help="Whether to use data augmentation or not for training data balancing")
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

        # Run Mismatch Labeler
        data = MismatchLabeler(data)()

        data.reset_index(drop=True, inplace=True)

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
        ].head(200000)
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

def train_model(train_data, hyper_params, use_aug=False, model_name='random_forest'):
    run = wandb.init(config=hyper_params)

    if model_name == 'distilbert':
        model = DistilbertPrivacyModel(**hyper_params)
    else:
        model = RandomForestPrivacyModel(**hyper_params)

    # #drop train examples with Label=1 and Body less than 4 words
    # train_data = train_data[~((train_data['Label'] == 1) & (train_data['Body'].str.split().str.len() < 4))]
    # train_data = train_data.reset_index(drop=True)

    if use_aug:
        augmentor = Augmentor()

        train_body, train_labels = augmentor(
            train_data['Body'].tolist(), 
            train_data['Label'].tolist(), 
            aug_label=1, 
            num_aug_per_label_1=9,
            shuffle=True
        )

        train_data = pd.DataFrame(
            {
                'Body': train_body,
                'Label': train_labels
            }
        )

    train_data.drop_duplicates(subset=['Body'], inplace=True)
    train_data.reset_index(drop=True, inplace=True)

    if model_name == 'distilbert':
        model.train(
            body=train_data['Body'], 
            label=train_data['Label'], 
            validation_size=0.2, 
            wandb=run
        )
    else:
        model.train(
            body=train_data['Body'], 
            label=train_data['Label'],
            wandb=run
        )
    

    # Restore the original stdout
    # sys.stdout = sys.__stdout__

    # Close the log file
    # log_file.close()
    return model

def test_model(train_data, sanity_data, gold_fraud_data, save_path):
    # Define a dictionary to store the f1 scores
    f1_scores = {}
    
    # Define a dictionary to store the predictions, true labels for each dataset
    # true_pred_map = {
    #     'train':{},
    #     'sanity':{},
    #     'gold_fraud':{}
    # }

    os.makedirs(os.path.join(save_path,'logs'), exist_ok=True)

    # Save the model and logs to the date folder
    model.save_model(os.path.join(save_path,'model'))

    train_data['Prediction'] = model.predict(body=train_data['Body'])
    evaluate_and_log(x=train_data['Body'].tolist(), y_true=train_data['Label'].tolist(), y_pred=train_data['Prediction'].tolist(), filename=os.path.join(save_path,'logs/train.log'), experiment=run, id = train_data['Mail-ID'].tolist())
    f1_scores['train'] = get_f1_score(y_true=train_data['Label'].tolist(), y_pred=train_data['Prediction'].tolist())

    sanity_data['Prediction'] = model.predict(body=sanity_data['Body'])
    evaluate_and_log(x=sanity_data['Body'].tolist(), y_true=sanity_data['Label'].tolist(), y_pred=sanity_data['Prediction'].tolist(), filename=os.path.join(save_path,'logs/sanity.log'), experiment=run, id = sanity_data['Mail-ID'].tolist())
    f1_scores['sanity'] = get_f1_score(y_true=sanity_data['Label'].tolist(), y_pred=sanity_data['Prediction'].tolist())

    gold_fraud_data['Prediction'] = model.predict(body=gold_fraud_data['Body'])
    evaluate_and_log(x=gold_fraud_data['Body'].tolist(), y_true=gold_fraud_data['Label'].tolist(), y_pred=gold_fraud_data['Prediction'].tolist(), filename=os.path.join(save_path,'logs/gold_fraud.log'), experiment=run, id = gold_fraud_data['Mail-ID'].tolist())
    f1_scores['gold_fraud'] = get_f1_score(y_true=gold_fraud_data['Label'].tolist(), y_pred=gold_fraud_data['Prediction'].tolist())

    #save mismatch data into a csv file
    mismatch_data = pd.concat(
        [
            train_data[train_data['Prediction'] != train_data['Label']],
            sanity_data[sanity_data['Prediction'] != sanity_data['Label']],
            gold_fraud_data[gold_fraud_data['Prediction'] != gold_fraud_data['Label']]
        ],
        axis=0,
        ignore_index=True
    )

    mismatch_data.to_csv(os.path.join(save_path,'logs/mismatch_data.csv'), index=False)

    return f1_scores

def dump_logs_to_wandb(hyper_params, f1_scores, save_path):
    # Log the hyperparameters and f1 scores to Weights and Biases
    all_params = {**hyper_params, **f1_scores}
    run.config.update(all_params)

    # Log the model to Weights and Biases
    model_path = os.path.join(save_path,'model')
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
    
    args.device = args.device if args.device != 'gpu' else 'cuda'
    
    if type(args.use_aug) == str:
        if args.use_aug.lower() == 'true':
            args.use_aug = True
        elif args.use_aug.lower() == 'false':
            args.use_aug = False
        else:
            raise ValueError("Invalid value for use_aug. Please enter True or False.")
    
    args.model_name = args.model_name.lower()
        
    # Define model hyperparameters
    if args.model_name == 'distilbert':
        hyper_params = {
            'num_labels': args.num_labels,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'device': args.device,
        }
    else:
        hyper_params = {
            'num_labels': args.num_labels,
            'n_estimators': args.n_estimators,
            'criterion': args.criterion,
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
    model = train_model(train_data, hyper_params, use_aug=args.use_aug, model_name = args.model_name)

    # Test the model
    f1_scores = test_model(train_data, sanity_data, gold_fraud_data, save_path)

    # Dump the logs to Weights and Biases
    dump_logs_to_wandb(hyper_params, f1_scores, save_path)

    # Close the Weights and Biases run
    run.finish()

