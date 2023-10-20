from typing import List, Optional
import os
import pandas as pd
import glob
import email
from concurrent.futures import ThreadPoolExecutor
import requests
import tarfile

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
        try_web: Optional[bool] = True,
    ) -> pd.DataFrame:
        """Load the Enron email data

        Note: 
            To run this locally
        
        Args:
            datapath (str, optional): Path to the Enron email data. Defaults to None.
            try_web (bool, optional): Try to download the data from the web if the data is not found locally. Defaults to True.

        Returns:
            email_df (pd.DataFrame): DataFrame containing the email data
        """

        self.datapath = datapath

        if self.datapath is None:
            self.datapath = config['data']['enron']

        if self.datapath.lower().startswith('http') or self.datapath.lower().startswith('www'):
            if os.path.exists(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    '../data/enron/maildir'
                )
            ):
                # If data exists in ../data/enron/, use the data directly
                self.datapath = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    '../data/enron/maildir'
                )
                pass
            
            else:
                #Download data from URL and proceed
                print('\x1b[4mLoadEnronData\x1b[0m: Downloading data from online source')
                
                #To note here, self.datapath is a web URL from where we download the data
                os.system(f"wget {self.datapath} -O /tmp/enron.tar.gz")

                self.datapath = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    '../data/enron/maildir'
                )

                #Since the path doesnt exist, make the folders
                os.makedirs(self.datapath)
                
                #Extract the tar.gz file
                with tarfile.open("/tmp/enron.tar.gz", "r:gz") as tar:
                    tar.extractall(self.datapath)
        
        if not os.path.exists(self.datapath):
            raise FileNotFoundError(f'\x1b[4mLoadEnronData\x1b[0m: Data not found at path: {self.datapath}')
        
        print(f'\x1b[4mLoadEnronData\x1b[0m: Loading data from path: {self.datapath}')
        
        #Load all file names
        # files = glob.glob(os.path.join(self.datapath,"/**/*."), recursive=True)
        files = self.collect_files_in_directory(self.datapath)
        
        print(f'\x1b[4mLoadEnronData\x1b[0m: Load Data Successful')

        # Get the email fields
        email_df = self.get_email_df(files)
        
        print('\x1b[4mLoadEnronData\x1b[0m: Data Successfully loaded into a DataFrame')
        
        return email_df
    
    def collect_files_in_directory(
            self,
            root_dir, 
            extension = '.'
        ) -> list[str]:
        """Collect all files in a directory

        Args:
            root_dir (str): Root directory
            extension (str, optional): Extension of the files to collect. Defaults to '.'.

        Returns:
            files (list): List of all files in the directory
        """

        files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.'):
                    file_path = os.path.join(dirpath, filename)
                    files.append(file_path)
        return files
    
    def process_email(
        self,
        file: str,
    ) -> dict:
        """Process the email data
        
        Args:
            file (str): Path to the email file

        Returns:
            email_fields (dict): Dictionary containing the email fields
        """

        email_fields = {}
        folder_user = file.split(self.datapath)[1].split('/')[0]
        folder_name = file.split(self.datapath)[1].split('/')[1]

        email_fields['Folder-User'] = folder_user
        email_fields['Folder-Name'] = folder_name

        with open(file, "rb") as binary_file:
            msg = email.message_from_binary_file(binary_file)

        # Extract fields from the email
        for field in msg.keys():
            email_fields[field] = msg[field]

        # Extract the email body
        email_fields['Body'] = msg.get_payload()

        return email_fields

    def get_email_df(
        self,
        files
    ) -> pd.DataFrame:
        """Get the email DataFrame

        Args:
            files (list): List of all files in the directory

        Returns:
            email_df (pd.DataFrame): DataFrame containing the email data
        """

        emails = []

        with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
            results = list(executor.map(self.process_email, files))
            emails.extend(results)

        return pd.DataFrame(emails)
