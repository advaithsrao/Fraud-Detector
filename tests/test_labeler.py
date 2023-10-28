import os
import sys
sys.path.append("..")

import pandas as pd
import pytest
from utils.data_fetch import LoadEnronData
from detector.labeler import EnronLabeler

@pytest.fixture
def dataframe():
    data_loader = LoadEnronData()
    data = data_loader(
        datapath = os.path.join(
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
    assert type(pipeline.get_phishing_model_annotation()) == pd.DataFrame
    assert type(pipeline.get_social_engineering_annotation()) == pd.DataFrame
    assert type(pipeline.get_labels()) == pd.DataFrame


