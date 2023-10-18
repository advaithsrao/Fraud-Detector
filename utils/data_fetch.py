from typing import List, Optional
import os
import pandas as pd
import glob
import email
from concurrent.futures import ThreadPoolExecutor

#read config.ini file
import configparser
config = configparser.ConfigParser()
config.read(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../config.ini'
    )
)


class PersonOfInterest:
    def __init__(
        self,
        name_list: Optional[List[str]] = None,
        email_list: Optional[List[str]] = None,
    ):
        """Class to operate with the person of interest data from config.ini file
        """
        self.poi = {}

        #read [person_of_interest_name] and [person_of_interest_email] section from config.ini file if not given explicitly
        if name_list is None:
            self.poi['names'] = config['person_of_interest.names']['names']
        else:
            self.poi['names'] = name_list
        
        if email_list is None:
            self.poi['emails'] = config['person_of_interest.emails']['emails']
        else:
            self.poi['emails'] = email_list
        
        #convert the values to lists
        self.poi['names'] = [name.strip() for name in self.poi['names'].split('&')]
        self.poi['emails'] = [email.strip() for email in self.poi['emails'].split('&')]
    
    def check_person_of_interest_name(
        self,
        name: str
    ):
        if name in self.poi['names']:
            return True
    
    def check_person_of_interest_email(
        self,
        email: str
    ):  
        if email in self.poi['emails']:
            return True
    
    def return_person_of_interest(
        self,
    ):
        return self.poi


class LoadEnronData:
    def __call__(
        self,
        datapath: Optional[str] = None,
    ):
        """Load the Enron email data

        Note: 
            To run this locally
        
        Args:
            datapath (str, optional): Path to the Enron email data. Defaults to None.

        Returns:
            email_df (pd.DataFrame): DataFrame containing the email data
        """
        self.datapath = datapath

        if self.datapath is None:
            self.datapath = config['data']['enron']
        
        # Get all the email files
        files = glob.glob(datapath + "/**/*.", recursive=True)

        # Get the email fields
        email_df = self.get_email_df(files)

        return email_df
    
    def process_email(
        self,
        file: str,
    ):
        email_fields = {}
        folder_user = file.split(self.datapath)[1].split('/')[0]
        folder_name = file.split(self.datapath)[1].split('/')[1]

        email_fields['Folder-User'] = folder_user
        email_fields['Folder_Name'] = folder_name

        with open(file, "rb") as binary_file:
            msg = email.message_from_binary_file(binary_file)

        # Extract fields from the email
        for field in msg.keys():
            email_fields[field] = msg[field]

        # Extract the email body
        email_fields['Body'] = msg.get_payload()

        # print(f'Done with user {folder_user} and folder {folder_name}')
        return email_fields

    def get_email_df(
        self,
        files
    ):
        emails = []

        with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
            results = list(executor.map(self.process_email, files))
            emails.extend(results)

        return pd.DataFrame(emails)
