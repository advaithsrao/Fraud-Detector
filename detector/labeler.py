from typing import Any
import sys
sys.path.append("..")

import json
import hashlib
import os
import pandas as pd
import re
import swifter
from mlflow.sklearn import load_model

from detector.preprocessor import Preprocessor
from detector.data_loader import LoadEnronData, PersonOfInterest
from utils.util_preprocessor import add_subject_to_body, convert_string_to_list
from utils.util_data_loader import download_file_from_google_drive

#read config.ini file
import configparser
config = configparser.ConfigParser()
config.read(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../config.ini'
    )
)


class EnronLabeler:
    """Class to label the enron data

    Args:
        data (pd.DataFrame): DataFrame containing the enron data
        needs_preprocessing (bool): True if the data needs to be preprocessed else False
        preprocessor (Preprocessor): Preprocessor object to preprocess the text
        person_of_interest (PersonOfInterest): PersonOfInterest object to check if person of interest is present in the text
        cfg (configparser.ConfigParser): ConfigParser object to read config.ini file
    
    Returns:
        data (pd.DataFrame): DataFrame containing the enron data with labeling updates
    """
    
    def __init__(
        self, 
        data: pd.DataFrame = None,
        needs_preprocessing: bool = False,
        preprocessor: Preprocessor = None,
        person_of_interest: PersonOfInterest = None,
        cfg: configparser.ConfigParser = None,
    ):
        
        self.data = data
        self.needs_preprocessing = needs_preprocessing
        self.preprocessor = preprocessor
        self.person_of_interest = person_of_interest
        self.config = cfg

        if self.data is None:
            self.data = LoadEnronData()
        
        if self.preprocessor is None:
            self.preprocessor = Preprocessor()
        
        if self.person_of_interest is None:
            self.person_of_interest = PersonOfInterest().return_person_of_interest()
        
        if self.config is None:
            self.config = config
        
        if 'Body' not in self.data.columns:
            raise ValueError('Body column not found in the data')
        
        if 'From' not in self.data.columns:
            raise ValueError('From column not found in the data')
        
        if 'To' not in self.data.columns:
            raise ValueError('To column not found in the data')
        
        if 'Cc' not in self.data.columns:
            raise ValueError('Cc column not found in the data')
        
        if 'emails' not in self.person_of_interest.keys():
            raise ValueError('Person of interest emails not found in the PersonOfInterest object')
        
        if 'names' not in self.person_of_interest.keys():
            raise ValueError('Person of interest names not found in the PersonOfInterest object')
        
        print(f'\x1b[4mEnronLabeler\x1b[0m: Initialized Successfully!')
        
    def __call__(
        self
    ) -> pd.DataFrame:
        
        """Call the Pipeline to label the enron data

        Returns:
            pd.DataFrame: DataFrame containing the enron data with labels
        """

        self.data.fillna(' ', inplace=True)
        print(f'\x1b[4mEnronLabeler\x1b[0m: NaN Values replaced with " "')

        self.data = self.concat_subject_body(self.data)
        print(f'\x1b[4mEnronLabeler\x1b[0m: Appended Subject to Body column')

        if self.needs_preprocessing:
            self.data['Body'] = self.data['Body'].swifter.apply(self.preprocessor)
            print(f'\x1b[4mEnronLabeler\x1b[0m: Preprocessed Body Column')
        
        self.data = self.convert_cc_to_list(self.data)
        print(f'\x1b[4mEnronLabeler\x1b[0m: Converted CC column to list of values')

        self.data = self.poi_present(self.data)
        print(f'\x1b[4mEnronLabeler\x1b[0m: POI Present column added')

        self.data = self.suspicious_folder(self.data)
        print(f'\x1b[4mEnronLabeler\x1b[0m: Suspicious Folder column added')
        
        self.data = self.check_sender_type(self.data)
        print(f'\x1b[4mEnronLabeler\x1b[0m: Check Sender Type column added')

        self.data = self.check_unique_mails_from_sender(self.data)
        print(f'\x1b[4mEnronLabeler\x1b[0m: Unique Mails from sender column added')
        print(f'\x1b[4mEnronLabeler\x1b[0m: Low Comm column added')

        self.data = self.contains_replies_forwards(self.data)
        print(f'\x1b[4mEnronLabeler\x1b[0m: Contains Reply Forwards column added')

        # self.data = self.get_url_count(self.data)
        # print(f'\x1b[4mEnronLabeler\x1b[0m: URL Count column added')

        self.data = self.get_phishing_model_annotation(self.data)
        print(f'\x1b[4mEnronLabeler\x1b[0m: Phishing Model Annotation column added')

        self.data = self.get_social_engineering_annotation(self.data)
        print(f'\x1b[4mEnronLabeler\x1b[0m: Social Engineering Annotation column added')

        self.data = self.get_labels(self.data)
        print(f'\x1b[4mEnronLabeler\x1b[0m: Fraud Labels added')

        return self.data

    def concat_subject_body(
        self,
        data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Concatenate subject and body columns
        
        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.
        
        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with subject column appended in the body column
        """

        if data is None:
            data = self.data
        
        data['Body'] = data.swifter.apply(
            lambda row:
                add_subject_to_body(
                    row['Subject'],
                    row['Body']
                ),
            axis = 1
        )

        return data

    def convert_cc_to_list(
        self,
        data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Convert the cc column to list in enron data

        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.

        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with cc column converted to list
        """
        
        if data is None:
            data = self.data
        
        data['Cc'] = data['Cc'].swifter.apply(lambda x: convert_string_to_list(x, sep = ',') if type(x) == str and ',' in x else x)
        
        return data
    
    def poi_present(
        self,
        data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Check if person of interest is present in the text

        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.

        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with new column `poi_present` -> True when poi is present in the text else False
        """

        if data is None:
            data = self.data

        data['POI-Present'] = data.swifter.apply(
            lambda row:
                True if row['To'] in self.person_of_interest['emails'] \
                    or (
                        type(row['Cc']) == list \
                        and \
                        bool(
                                [
                                    email for email in row['Cc'] if email in self.person_of_interest['emails']
                                ]
                        ) \
                    ) \
                    or (
                        type(row['Cc']) == str \
                        and \
                        row['Cc'] in self.person_of_interest['emails']
                    ) \
                    else False,
            axis=1
        )

        return data
    
    def suspicious_folder(
        self,
        data: pd.DataFrame = None,
        config: configparser.ConfigParser = None,
    ) -> pd.DataFrame:
        """Check if the email is in the suspicious folder

        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.

        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with new column `suspicious_folder` -> True when email is in the suspicious folder else False
        """

        if data is None:
            data = self.data

        if config is None:
            config = self.config
        
        suspicious_folders = config['folders.possible_fraud']['folders'].split(' & ')
        suspicious_folders = [folder.strip() for folder in suspicious_folders]

        data['Suspicious-Folders'] = data['Folder-Name'].swifter.apply(lambda x: True if x in suspicious_folders else False)

        return data

    def check_sender_type(
        self,
        data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Check if the sender is internal or external

        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.

        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with new column `sender_type` -> internal or external
        """

        if data is None:
            data = self.data

        data['Sender-Type'] = data['From'].swifter.apply(
            lambda x: 'Internal' \
                if '@' in x \
                    and x.split('@')[1].endswith('enron.com')
                else 'External'
        )

        return data
    
    def check_unique_mails_from_sender(
        self,
        data: pd.DataFrame = None,
        unique_mails_dict: dict[str, int] = None,
    ) -> pd.DataFrame:
        """
        """

        if data is None:
            data = self.data
        
        if unique_mails_dict is None:
            dict_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    '../resources/unique_mails_from_sender.json'
            )

            if not os.path.exists(dict_path):
                raise ValueError('unique_mails_from_sender.json not found in the resources folder')

            with open(dict_path, 'r') as f:
                unique_mails_dict = json.load(f)
        
        data['Unique-Mails-From-Sender'] = data['From'].swifter.apply(
            lambda x: unique_mails_dict[x] if x in unique_mails_dict.keys() else 0
        )
        
        data['Low-Comm'] = data['Unique-Mails-From-Sender'].swifter.apply(
            lambda x: True if x <= 4 else False
        )

        return data
    
    def contains_replies_forwards(
        self,
        data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Check if the email body contains replies and/or forwards

        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.

        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with new column 'Contains-Reply-Forwards' 
            -> True when email body contains replies and/or forwards else False
        """

        if data is None:
            data = self.data

        reply_patterns = self.config.get('labeler.mismatch','replies')
        pattern = fr'\b(?:{reply_patterns})\b'
        
        data['Contains-Reply-Forwards'] = data['Body'].swifter.apply(
            lambda x: bool(
                re.search(
                    pattern, 
                    x, 
                    flags=re.IGNORECASE
                )
            )
        )

        return data
    
    def get_url_count(
        self,
        data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Get the url count in the email body

        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.

        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with new column 'URL-Count' 
            -> Count of URLs in the email body
        """

        if data is None:
            data = self.data
        
        data['URL-Count'] = data['Body'].swifter.apply(
            lambda x: len(re.findall(r'<URL>', x))
        )

        return data
    
    def get_prediction_on_enron(
        self,
        data: pd.Series = None,
        model = None,
    ) -> int:
        """Predict a model on a row of enron data

        Args:
            data (pd.Series): Row of enron data
            model (sklearn.model): Model to predict on the row of enron data

        Returns:
            int: Prediction of the model
        """
        
        if data is None:
            data = self.data
        
        if model == None:
            raise ValueError('model not provided')

        if data['Sender-Type'] == 'External' and data['Contains-Reply-Forwards'] == False and data['Low-Comm'] == True:
            pred = model.predict([data['Body']])[0]
            return pred
        return 0
    
    def get_phishing_model_annotation(
        self,
        data: pd.DataFrame = None,
        model_path: str = "../resources/Phishing-Model/"
    ) -> pd.DataFrame:
        """Get phishing model prediction

        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.
            model_path (str, optional): Path to the phishing model. Defaults to "../resources/Phishing-Model/".

        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with new column 'Phishing-Model-Prediction' 
            -> Prediction of the phishing model
        """

        if data is None:
            data = self.data
        
        if model_path == '':
            raise ValueError('Phishing model_path not provided')
        
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            model_path
        )

        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
            #download files from google drive
            ids = convert_string_to_list(config['model.annotator.phishing']['ids'])
            file_names = ['conda.yaml', 'MLmodel', 'model.pkl', 'python_env.yaml', 'requirements.txt']

            for id, file_name in zip(ids, file_names):
                download_file_from_google_drive(id, os.path.join(model_path, file_name))
        
        try:
            model = load_model(model_path)
        except:
            raise ValueError(f'Phishing Model not found in {model_path}')

        data['Phishing-Annotation'] = data.swifter.apply(lambda x: self.get_prediction_on_enron(data = x, model = model), axis = 1)

        return data

    def get_social_engineering_annotation(
        self,
        data: pd.DataFrame = None,
        model_path: str = "../resources/SocEngg-Model/"
    ):
        """Get Social Engineering model prediction

        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.
            model_path (str, optional): Path to the social engineering model. Defaults to "../resources/SocEngg-Model/".

        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with new column 'Social-Engineering-Model-Prediction' 
            -> Prediction of the social engineering model
        """

        if data is None:
            data = self.data
        
        if model_path == '':
            raise ValueError('Social Engineering model_path not provided')
        
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            model_path
        )

        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
            #download files from google drive
            ids = convert_string_to_list(config['model.annotator.social_engineering']['ids'])
            file_names = ['conda.yaml', 'MLmodel', 'model.pkl', 'python_env.yaml', 'requirements.txt']

            for id, file_name in zip(ids, file_names):
                download_file_from_google_drive(id, os.path.join(model_path, file_name))
        
        try:
            model = load_model(model_path)
        except:
            raise ValueError(f'Social Engineering Model not found in {model_path}')

        data['SocEngg-Annotation'] = data.swifter.apply(lambda x: self.get_prediction_on_enron(data = x, model = model), axis = 1)

        return data
    
    def get_labels(
        self,
        data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Get the label of the email based on heuristics

        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.

        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with new column 'Label' 
            -> Label of the email
        """

        if data is None:
            data = self.data

        data['Label'] = data.swifter.apply(
            lambda row:
                1 if row['Phishing-Annotation'] == 1 or row['SocEngg-Annotation'] == 1 else 0,
            axis = 1
        )

        return data

class MismatchLabeler:
    """Class to relabel the mismatch examples from our dataset

    Args:
        data (pd.DataFrame): DataFrame
        cfg (configparser.ConfigParser): ConfigParser object to read config.ini file
    
    Returns:
        data (pd.DataFrame): DataFrame containing the relabeled data with labeling updates
    """
    
    def __init__(
        self, 
        data: pd.DataFrame = None,
        cfg: configparser.ConfigParser = None,
    ):
        
        self.data = data
        self.config = cfg

        if self.data is None:
            raise ValueError('data not provided')
        
        if self.config is None:
            self.config = config
    
    def __call__(
        self
    ) -> pd.DataFrame:
        
        """Call the Pipeline to label the enron data

        Returns:
            pd.DataFrame: DataFrame containing the enron data with labels
        """

        self.data = self.drop_by_length(self.data)
        print(f'\x1b[4mMismatchLabeler\x1b[0m: Dropped examples with body length less than 4 words and more than 600 words')

        self.data = self.drop_by_pattern(self.data)
        print(f'\x1b[4mMismatchLabeler\x1b[0m: Dropped examples with body containing the given pattern')

        self.data = self.relabel_marketing_frauds(self.data)
        print(f'\x1b[4mMismatchLabeler\x1b[0m: Relabeled marketing examples with label 1 to label 0 using marketing keywords')

        return self.data
    
    def drop_by_length(
        self,
        data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Drop the fraud examples with body length less than 4 words and more than 600 words

        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.

        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with examples dropped
        """

        if data is None:
            data = self.data
        
        drop_threshold = self.config.get('labeler.mismatch','drop_threshold')
        min_length, max_length = convert_string_to_list(drop_threshold, sep = '&')
        min_length, max_length = int(min_length), int(max_length)

        data = data[~((data['Label'] == 1) & (data['Body'].str.split().str.len() < min_length))]
        data = data[~((data['Label'] == 1) & (data['Body'].str.split().str.len() > max_length))]
        
        return data
    
    def drop_by_pattern(
        self,
        data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Drop the fraud examples with body containing the given pattern

        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.

        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with examples dropped
        """

        if data is None:
            data = self.data
        
        patterns = [
            r'' + config.get('labeler.mismatch', 'best_regards'),
            r'' + config.get('labeler.mismatch', 'sincerely'),
            r'' + config.get('labeler.mismatch', 'regards'),
            r'' + config.get('labeler.mismatch', 'your_sincerely'),
            r'' + config.get('labeler.mismatch', 'yours_sincerely'),
            r'' + config.get('labeler.mismatch', 'yours_truly'),
            r'' + config.get('labeler.mismatch', 'yours_faithfully'),
            r'' + config.get('labeler.mismatch', 'thanks'),
            r'' + config.get('labeler.mismatch', 'thank_you'),
            r'' + config.get('labeler.mismatch', 'message_id'),
            r'' + config.get('labeler.mismatch', 'from'),
            r'' + config.get('labeler.mismatch', 'sent'),
            r'' + config.get('labeler.mismatch', 'to'),
            r'' + config.get('labeler.mismatch', 'cc'),
            r'' + config.get('labeler.mismatch', 'undelivery'),
            r'' + config.get('labeler.mismatch', 'undeliverable'),
            r'' + config.get('labeler.mismatch', 'missed_reply')
        ]

        # Create a temporary column without Subject
        data['Temp_Body'] = data.swifter.apply(lambda row: row['Body'].replace(row['Subject'], '') if pd.notna(row['Subject']) else row['Body'], axis=1)

        combined_pattern = '|'.join(f'(?:^|^\s|^>|^ >)(?: |){pattern}' for pattern in patterns)

        # Filter out rows where Label is 1 and any pattern matches
        data = data[~((data['Label'] == 1) & data['Temp_Body'].str.contains(combined_pattern, case=False, regex=True))]

        # Drop the temporary column
        data = data.drop(columns=['Temp_Body'])

        return data
    
    def relabel_marketing_frauds(
        self,
        data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Relabel the marketing examples with label 1 to label 0 using marketing keywords

        Args:
            data (pd.DataFrame, optional): DataFrame containing the enron data. Defaults to None.

        Returns:
            data (pd.DataFrame): DataFrame containing the enron data with new column 'Label' 
            -> Label of the email
        """
        
        if data is None:
            data = self.data
        
        marketing_keywords = self.config.get('labeler.mismatch','marketing')
        
        data.loc[
            (data['Label'] == 1) & \
            data['Body'].str.contains(
                marketing_keywords,
                case=False, regex=True
            ), 
            'Label'
        ] = 0
    
        return data
    
