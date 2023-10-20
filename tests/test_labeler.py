import os
import sys
sys.path.append("..")

import pandas as pd
import pytest
from utils.data_fetch import LoadEnronData
from detector.labeler import EnronPipeline

@pytest.fixture
def dataframe():
    data_loader = LoadEnronData()
    return data_loader(
        datapath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../resources/enron/sample'
        ),
        try_web=False
    )

def test_enron_pipeline():
    pipeline = EnronPipeline(dataframe)
    result = pipeline()
    assert type(result) == pd.DataFrame


