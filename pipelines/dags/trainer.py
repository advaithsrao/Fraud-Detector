import sys
sys.path.append('../..')

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
import os
from detector.data_loader import LoadEnronData, LoadPhishingData, LoadSocEnggData
from detector.labeler import EnronLabeler
from detector.modeler import RobertaModel
from detector.preprocessor import Preprocessor
from utils.util_modeler import evaluate_and_log, get_f1_score

import wandb

#read config.ini file
import configparser

config = configparser.ConfigParser()
config.read(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../config.ini'
    )
)

wandbdict = {
    'key': config['wandb']['api_key'],
    'entity': config['wandb']['entity'],
    'project': config['wandb']['project']
}

wandb.login(**wandbdict)
run = wandb.init()

# Define model hyperparameters
hyper_params = {
    'num_labels' : 2,
    'num_epochs' : 1,
    'batch_size' : 10
}

# Define your default_args and DAG object
default_args = {
    'owner': 'fraud-detector',
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
}


#If we need the pipeline to run on a schedule, we can set the schedule_interval
dag = DAG(
    'trainer',
    default_args=default_args,
    description='Fraud-Detector pipeline to load, preprocess, label, and train the model',
    schedule_interval=None,  # Set the schedule as needed
)

# Define a variable to store the trained model
model = None

# Get the current date
date = datetime.now().strftime("%Y-%m-%d")

# Create date folder in /tmp
os.makedirs(f'/tmp/{date}', exist_ok=True)

def load_data(**kwargs):
    # Loader
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
    
    ti = kwargs['ti']
    ti.push(key='data', value=data)

def label_and_preprocess_data(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='load_data')

    if not os.path.exists(
        os.path.join(
            os.path.dirname(__file__), 
            '../data/fraud_detector_data.csv'
        )
    ):
        #Run Enron Labeler
        data['LoadEnronData'] = EnronLabeler(data['LoadEnronData'], needs_preprocessing=True)()

        #Preprocess the other 2 datasets
        data['LoadPhishingData']['Body'] = data['LoadPhishingData']['Body'].swifter.apply(Preprocessor())
        data['LoadSocEnggData']['Body'] = data['LoadSocEnggData']['Body'].swifter.apply(Preprocessor())

        #Concatenate the 3 data sources into 1
        data = pd.concat(
            [
                df for df in data.values()
            ], 
            axis=0, 
            ignore_index=True
        )

def data_split(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='label_and_preprocess_data')

    if not os.path.exists(
        os.path.join(
            os.path.dirname(__file__), 
            '../data/fraud_detector_data.csv'
        )
    ):
        #For gold_fraud_set, Take first 500 emails from Phishing Data and 500 emails from Social Engineering Data
        gold_fraud = pd.concat(
            [
                data[data['Source'] == 'Phishing Data'][data['Label'] == 1].head(500),
                data[data['Source'] == 'Social Engineering Data'][data['Label'] == 1].head(500)
            ],
            axis=0,
            ignore_index=True
        )
        gold_fraud['Split'] = 'Gold Fraud'

        #For sanity_set, Take first 5000 emails with Sender-Type = 'Internal'
        sanity = data[
            (data['Sender-Type'] == 'Internal') & (data['Source'] == 'Enron Data')
        ].head(5000)
        sanity['Split'] = 'Sanity'

        #For train_set, Take all data not in gold_fraud_set and sanity_set
        train = data[
            ~data['Mail-ID'].isin(gold_fraud['Mail-ID']) & ~data['Mail-ID'].isin(sanity['Mail-ID'])
        ]
        train['Split'] = 'Train'
    else:
        train = data[data['Split'] == 'Train'].head(20)
        gold_fraud = data[data['Split'] == 'Gold Fraud'].head(20)
        sanity = data[data['Split'] == 'Sanity'].head(20)
    
    ti.xcom_push(key='train_data', value=train)
    ti.xcom_push(key='sanity_data', value=sanity)
    ti.xcom_push(key='gold_fraud_data', value=gold_fraud)

def train_model(**kwargs):
    ti = kwargs['ti']
    train = ti.xcom_pull(key='train_data', task_ids='data_split')

    run = wandb.init(config=hyper_params)
    
    model = RobertaModel(**hyper_params)

    os.makedirs(f'/tmp/{date}/logs', exist_ok=True)

    # Define a log file path
    log_filename = f"/tmp/{date}/logs/model_training.log"

    # Create or open the log file in write mode
    log_file = open(log_filename, "w")

    # Redirect stdout to the log file
    sys.stdout = log_file

    # Call your code that produces output
    model.train(body=train['Body'], label=train['Label'], validation_size=0.2, wandb=run)

    # Restore the original stdout
    sys.stdout = sys.__stdout__

    # Close the log file
    log_file.close()

    ti.xcom_push(key='hyper_params', value=hyper_params)

def test_model(**kwargs):
    ti = kwargs['ti']
    train = ti.xcom_pull(key='train_data', task_ids='data_split')
    sanity = ti.xcom_pull(key='sanity_data', task_ids='data_split')
    gold_fraud = ti.xcom_pull(key='gold_fraud_data', task_ids='data_split')

    f1_scores = {}

    # Save the model and logs to the date folder
    model.save_model(f'/tmp/{date}/model')

    train_labels = train['Label'].tolist()
    train_preds = model.predict(body = train['Body'])
    evaluate_and_log(x = train['Body'], y_true = train_labels, y_pred = sanity_preds, filename = '/tmp/{date}/logs/train.log')
    f1_scores['train'] = get_f1_score(y_true = train_labels, y_pred = train_preds)

    sanity_labels = sanity['Label'].tolist()
    sanity_preds = model.predict(body = sanity['Body'])
    evaluate_and_log(x = sanity['Body'], y_true = sanity_labels, y_pred = sanity_preds, filename = '/tmp/{date}/logs/sanity.log')
    f1_scores['sanity'] = get_f1_score(y_true = sanity_labels, y_pred = sanity_preds)

    gold_fraud_labels = gold_fraud['Label'].tolist()
    gold_fraud_preds = model.predict(body = gold_fraud['Body'])
    evaluate_and_log(x = gold_fraud['Body'], y_true = gold_fraud_labels, y_pred = gold_fraud_preds, filename = '/tmp/{date}/logs/gold_fraud.log')
    f1_scores['gold_fraud'] = get_f1_score(y_true = gold_fraud_labels, y_pred = gold_fraud_preds)

    ti.xcom_push(key='f1_scores', value=f1_scores)
    ti.xcom_push(key='train_preds', value=train_preds)
    ti.xcom_push(key='sanity_preds', value=sanity_preds)
    ti.xcom_push(key='gold_fraud_preds', value=gold_fraud_preds)

def dump_logs_to_wandb(**kwargs):
    ti = kwargs['ti']
    hyper_params = ti.xcom_pull(key='hyper_params', task_ids='train_model')
    f1_scores = ti.xcom_pull(key='f1_scores', task_ids='test_model')

    train_labels = ti.xcom_pull(key='train_labels', task_ids='test_model')
    train_preds = ti.xcom_pull(key='train_preds', task_ids='test_model')

    sanity_labels = ti.xcom_pull(key='sanity_labels', task_ids='test_model')
    sanity_preds = ti.xcom_pull(key='sanity_preds', task_ids='test_model')

    gold_fraud_labels = ti.xcom_pull(key='gold_fraud_labels', task_ids='test_model')
    gold_fraud_preds = ti.xcom_pull(key='gold_fraud_preds', task_ids='test_model')

    # Log the hyperparameters and f1 scores to Weights and Biases
    all_params = {**hyper_params, **f1_scores}
    run.config.update(all_params)

    # Log the model to Weights and Biases
    model_path = f'/tmp/{date}/model'  # Update with your model path
    run.log_model(name="fraud-detector-model", path=model_path)

    # Log the log files to Weights and Biases
    log_files = [
        f'/tmp/{date}/logs/train.log',
        f'/tmp/{date}/logs/sanity.log',
        f'/tmp/{date}/logs/gold_fraud.log',
    ]
    for log_file in log_files:
        run.log_artifact(log_file, name=log_file)

    # Log confusion matrices
    run.log({
        "train_confusion_matrix": wandb.plot.confusion_matrix(train_labels, train_preds),
        "sanity_confusion_matrix": wandb.plot.confusion_matrix(sanity_labels, sanity_preds),
        "gold_fraud_confusion_matrix": wandb.plot.confusion_matrix(gold_fraud_labels, gold_fraud_preds),
    })

    # Finish the Weights and Biases run
    run.finish()


# Create tasks for each step
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,  # Pass the context to the callable function
    dag=dag,
)

label_and_preprocess_data_task = PythonOperator(
    task_id='label_data',
    python_callable=label_and_preprocess_data,
    provide_context=True,  # Pass the context to the callable function
    dag=dag,
)

data_split_task = PythonOperator(
    task_id='data_split',
    python_callable=data_split,
    provide_context=True,  # Pass the context to the callable function
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,  # Pass the context to the callable function
    dag=dag,
)

test_model_task = PythonOperator(
    task_id='predict_model',
    python_callable=test_model,
    provide_context=True,  # Pass the context to the callable function
    dag=dag,
)

dump_logs_to_wandb_task = PythonOperator(
    task_id='dump_logs_to_comet',
    python_callable=dump_logs_to_wandb,
    provide_context=True,  # Pass the context to the callable function
    dag=dag,
)

# Set task dependencies
load_data_task >> label_and_preprocess_data_task >> data_split_task >> train_model_task >> test_model_task >> dump_logs_to_wandb_task

if __name__ == "__main__":
    dag.cli()
