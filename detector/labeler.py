import sys
from typing import Any
sys.path.append("..")

import json
import os
import pandas as pd
import re
import swifter
from mlflow.sklearn import load_model

from utils.data_fetch import LoadEnronData, PersonOfInterest
from utils.cleanup import Preprocessor, add_subject_to_body, convert_string_to_list

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

        data['Contains-Reply-Forwards'] = data['Body'].swifter.apply(
            lambda x: True \
                if \
                    'Re:' in x \
                    or \
                    'RE:' in x \
                    or \
                    'Fw:' in x \
                    or \
                    'FW:' in x \
                    or \
                    'Fwd:' in x \
                    or \
                    'FWD:' in x \
                else False
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
        model_path: str = '',
    ) -> int:
        """Predict a model on a row of enron data

        Args:
            data (pd.Series): Row of enron data
            model_path (str): Path to the model

        Returns:
            int: Prediction of the model
        """
        
        if data is None:
            data = self.data
        
        if model_path == '':
            raise ValueError('model_path not provided')
        elif not os.path.exists(model_path):
            raise ValueError(f'{model_path} not found')
        
        model = load_model(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                model_path
            )
        )

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

        data['Phishing-Annotation'] = data.swifter.apply(lambda x: self.get_prediction_on_enron(data = x, model_path=model_path), axis = 1)

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

        data['SocEngg-Annotation'] = data.swifter.apply(lambda x: self.get_prediction_on_enron(data = x, model_path=model_path), axis = 1)

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