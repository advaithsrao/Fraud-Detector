import pandas as pd
import glob
import email
from concurrent.futures import ThreadPoolExecutor

#read config.ini file
import configparser
config = configparser.ConfigParser()
config.read('../config.ini')


class LoadEnronData:
    def __call__(
        self,
        datapath: str | None = None,
    ):
        """Load the Enron email data

        Note: 
            To run this, please specify the local path to enron dataset in config.ini. 
            Download path for enron dataset: https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
        
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
