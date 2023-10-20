import os
import sys
sys.path.append("..")

import pandas as pd
import pytest
from utils.cleanup import Preprocessor
from utils.data_fetch import PersonOfInterest, LoadEnronData


@pytest.fixture
def example():
    return """
    \n\n>  -----Original Message-----
    \n> From: \tHara, Kathy
    \n> Sent:\tMonday, April 09, 2001 11:53
    \n> To:\tMark Hackney (E-mail)
    \n> Cc:\tAllred, Penny; Cimino, Tony; Fewel, George; Holland, Kevin; Johnson,
    \n> Rob; Pearson, Tom; Rozelle, Dana; Begalman, Buppha; Downing, Staci;
    \n> \'Heather Bare\'; Locke, Kathy
    \n> Subject:\tBCHA Automatic Denial/Approval
    \n>\n> Mark
    \n>\n> We have been told by one of our Transmission Provider\'s that they do not
    \n> need to give us an OASIS number until half-past.  If we wait until
    \n> half-past to receive a valid oasis number, we cannot avoid launching late
    \n> tags.  I think that this places too much pressure on the merchant.
    \n>\n> We are also encountering problems with BC Hydro\'s automatic
    \n> approval/denial software.  What happens if a VALID tag is denied in the
    \n> "No Tag, No Flow" period, the control are cannot withdraw the denial, and
    \n> it is too late to launch another tag?  Which entity takes responsibility
    \n> for inadvertents and schedule cuts?
    \n>\n> I would like to get some of the timing issues resolved prior to
    \n> implementing "No Tag, No Flow."  The problems seem to be isolated, but it
    \n> only takes a single entity to create huge problems for everyone involved.
    \n>\n>\n> Thanks,
    \n> Kathy Hara
    """

def test_preprocesor(example):
    preprocess = Preprocessor()
    result = preprocess(example)
    
    #remove new lines
    assert '\n' not in result
    
    #remove specific patterns
    assert '-+Original Message-+' not in result
    assert 'From:' not in result
    assert 'Sent:' not in result
    assert 'To:' not in result
    assert 'Cc:' not in result
    assert 'Subject:' not in result

    #remove multiple whitespace
    assert '  ' not in result

def test_person_of_interest():
    poi = PersonOfInterest()
    assert type(poi.return_person_of_interest()) == dict
    assert type(poi.return_person_of_interest()['names']) == list
    assert type(poi.return_person_of_interest()['emails']) == list

    assert poi.check_person_of_interest_name('Lay, Kenneth') == True
    assert poi.check_person_of_interest_email('kenneth_lay@enron.net') == True

def test_load_enron_data():
    data_loader = LoadEnronData(
        datapath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../resources/enron/sample'
        ),
        try_web=False
    )
    
    data = data_loader()
    assert type(data) == pd.DataFrame

if __name__ == "__main__":
    pytest.main()
