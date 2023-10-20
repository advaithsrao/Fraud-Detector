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
    assert type(pipeline()) == pd.DataFrame

    #individual functions
    assert type(pipeline.poi_present()) == pd.DataFrame
    assert type(pipeline.suspicious_folder()) == pd.DataFrame
    assert type(pipeline.check_sender_type()) == pd.DataFrame
    assert type(pipeline.check_unique_mails_from_sender()) == pd.DataFrame


