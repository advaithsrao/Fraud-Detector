import sys
from typing import Any
sys.path.append("..")

import json
import os
import pandas as pd
from utils.data_fetch import LoadEnronData, PersonOfInterest
from utils.cleanup import Preprocessor

#read config.ini file
import configparser
config = configparser.ConfigParser()
config.read(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../config.ini'
    )
)


class EnronPipeline:
    """Class to label the enron data

    Args:
        data (pd.DataFrame): DataFrame containing the enron data
        preprocessor (Preprocessor): Preprocessor object to preprocess the text
        person_of_interest (PersonOfInterest): PersonOfInterest object to check if person of interest is present in the text
        cfg (configparser.ConfigParser): ConfigParser object to read config.ini file
    
    Returns:
        data (pd.DataFrame): DataFrame containing the enron data with labeling updates
    """
    
    def __init__(
        self, 
        data: pd.DataFrame | None = None,
        preprocessor: Preprocessor | None = None,
        person_of_interest: PersonOfInterest | None = None,
        cfg: configparser.ConfigParser | None = None,
    ):
        
        self.data = data
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
        
        self.data['Body'] = self.data['Body'].apply(self.preprocessor)
        self.data['Cc'] = self.data['Cc'].apply(self.convert_cc_to_list)
        
    def __call__(
        self
    ) -> pd.DataFrame:
        
        """Call the Pipeline to label the enron data

        Returns:
            pd.DataFrame: DataFrame containing the enron data with labels
        """
        
        self.data = self.poi_present(self.data)
        self.data = self.suspicious_folder(self.data)
        self.data = self.check_sender_type(self.data)
        self.data = self.check_unique_mails_from_sender(self.data)
        return self.data

    def convert_cc_to_list(
        self
    ) -> list[str] | None:
        """Convert the cc column to list

        Args:
            text (str): text to convert to list

        Returns:
            list[str] | None: list of cc emails
        """
        
        if type(text) != str:
            return text
        
        text = self.preprocessor(text)
        text = text.split(',')
        return [item.strip() for item in text]
    
    def poi_present(
        self,
        data: pd.DataFrame | None = None,
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
        data: pd.DataFrame | None = None,
        config: configparser.ConfigParser | None = None,
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
        data: pd.DataFrame | None = None,
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
        data: pd.DataFrame | None = None,
        unique_mails_dict: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        """
        """

        if data is None:
            data = self.data
        
        if unique_mails_dict is None:
            try:
                with open(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        '../resources/unique_mails_from_sender.json'
                    ), 
                    'w'
                ) as f:
                    unique_mails_dict = json.load(f)
            except Exception:
                raise ValueError('unique_mails_from_sender.json not found in the resources folder')
        
        data['Unique-Mails-From-Sender'] = data['From'].apply(
            lambda x: unique_mails_dict[x] if x in unique_mails_dict.keys() else 0
        )
        
        data['Low-Comm'] = data['Unique-Mails-From-Sender'].apply(
            lambda x: True if x <= 5 else False
        )
        
        return data
    
