import sys
sys.path.append("..")

import os
# import random
import pandas as pd
import pytest

from detector.data_loader import LoadEnronData
from detector.labeler import EnronLabeler, MismatchLabeler
from utils.util_data_loader import sha256_hash
from utils.util_preprocessor import convert_string_to_list

#read config.ini file
import configparser
config = configparser.ConfigParser()
config.read(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../config.ini'
    )
)

@pytest.fixture
def dataframe():
    data_loader = LoadEnronData()

    data = data_loader(
        localpath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../resources/enron/sample'
        ),
        try_web=False
    )
    
    return data  

def test_enron_labeler(dataframe):
    pipeline = EnronLabeler(dataframe)
    # assert type(pipeline()) == pd.DataFrame

    #individual functions
    assert type(pipeline.concat_subject_body()) == pd.DataFrame
    assert type(pipeline.convert_cc_to_list()) == pd.DataFrame
    assert type(pipeline.poi_present()) == pd.DataFrame
    assert type(pipeline.suspicious_folder()) == pd.DataFrame
    assert type(pipeline.check_sender_type()) == pd.DataFrame
    assert type(pipeline.check_unique_mails_from_sender()) == pd.DataFrame
    assert type(pipeline.contains_replies_forwards()) == pd.DataFrame
    assert type(pipeline.get_url_count()) == pd.DataFrame
    # assert type(pipeline.get_prediction_on_enron()) == ValueError
    # assert type(pipeline.get_phishing_model_annotation()) == pd.DataFrame
    # assert type(pipeline.get_social_engineering_annotation()) == pd.DataFrame
    # assert type(pipeline.get_labels()) == pd.DataFrame

def test_mismatch_labeler(dataframe):
    idx_of_body = dataframe.columns.tolist().index('Body')
    dataframe.iloc[0, idx_of_body] = 'Best Regards' + dataframe.iloc[0, idx_of_body]
    dataframe.iloc[1, idx_of_body] = '>From:' + dataframe.iloc[1, idx_of_body]
    dataframe.iloc[2, idx_of_body] = dataframe.iloc[2, idx_of_body] + 'Unsubscribe'
    dataframe.iloc[3, idx_of_body] = dataframe.iloc[3, idx_of_body] + 'update your preferences'
    dataframe['Label'] = [1] * len(dataframe)

    
    drop_threshold = config.get('labeler.mismatch','drop_threshold')
    min_length, max_length = convert_string_to_list(drop_threshold, sep = '&')
    min_length, max_length = int(min_length), int(max_length)

    pipeline = MismatchLabeler(dataframe)
    # assert type(pipeline()) == pd.DataFrame

    data = pipeline()
    
    assert type(data) == pd.DataFrame

    #drop by length function
    assert len(data[((data['Label'] == 1) & (data['Body'].str.split().str.len() < min_length))]) == 0
    assert len(data[((data['Label'] == 1) & (data['Body'].str.split().str.len() > max_length))]) == 0

    #drop by pattern function
    assert len(data[((data['Label'] == 1) & data['Body'].str.contains('(?:^|^\s|^>|^ >)(?: |)best regards', case=False, regex=True))]) == 0
    assert len(data[((data['Label'] == 1) & data['Body'].str.contains('(?:^|^\s|^>|^ >)(?: |)from:', case=False, regex=True))]) == 0

    #relabel marketing function
    assert len(data[((data['Label'] == 1) & data['Body'].str.contains('unsubscribe', case=False, regex=True))]) == 0

if __name__ == "__main__":
    pytest.main()
