import sys
from typing import Any
sys.path.append("..")

import json
import os
import pandas as pd
from utils.data_fetch import LoadEnronData, PersonOfInterest
from utils.cleanup import Preprocessor, add_subject_to_body
import re

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

        if 'Subject' in self.data.columns:
            self.data['Body'] = self.data.apply(
                lambda row:
                    add_subject_to_body(
                        row['Subject'],
                        row['Body']
                    ),
                axis = 1
            )
            print(f'\x1b[4mEnronLabeler\x1b[0m: Appended Subject to Body column')
        
        if self.needs_preprocessing:
            self.data['Body'] = self.data['Body'].apply(self.preprocessor)
            print(f'\x1b[4mEnronLabeler\x1b[0m: Preprocessed Body Column')

        #if Cc column is a string and not (None or list), convert it to list of strings
        self.data['Cc'] = self.data['Cc'].apply(lambda x: self.convert_cc_to_list(x) if type(x) == str else x)
        
    def __call__(
        self
    ) -> pd.DataFrame:
        
        """Call the Pipeline to label the enron data

        Returns:
            pd.DataFrame: DataFrame containing the enron data with labels
        """
        
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

        self.data = self.get_url_count(self.data)
        print(f'\x1b[4mEnronLabeler\x1b[0m: URL Count column added')

        return self.data

    def convert_cc_to_list(
        self,
        text: str,
    ) -> list[str]:
        """Convert the cc column to list

        Args:
            text (str): text to convert to list

        Returns:
            list[str] | None: list of cc emails
        """
        
        text = self.preprocessor(text)
        text = text.split(',')
        return [item.strip() for item in text]
    
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

        data['POI-Present'] = data.apply(
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

        data['Suspicious-Folders'] = data['Folder-Name'].apply(lambda x: True if x in suspicious_folders else False)

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

        data['Sender-Type'] = data['From'].apply(
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
        
        data['Unique-Mails-From-Sender'] = data['From'].apply(
            lambda x: unique_mails_dict[x] if x in unique_mails_dict.keys() else 0
        )
        
        data['Low-Comm'] = data['Unique-Mails-From-Sender'].apply(
            lambda x: True if x <= 5 else False
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

        data['Contains-Reply-Forwards'] = data['Body'].apply(
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
        
        data['URL-Count'] = data['Body'].apply(
            lambda x: len(re.findall(r'<URL>', x))
        )

        return data
    
