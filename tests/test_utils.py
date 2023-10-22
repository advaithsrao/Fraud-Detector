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

    \n\nHere is the project a little more refined. This was prepared by\nJeff Beale my partner in the project.\n\n\n---------- Original Message ----------------------------------\nFrom: Jeff Beale <JPBeale@ch-iv.com>\nDate: Wed, 09 May 2001 14:13:15 -0400\n\n<html>\n<font size=3D3 color=3D"#0000FF">Dear Greg, Eric, Addison:<br>\n<br>\n</font><font color=3D"#0000FF">Per my discussion with Bill Perkins\nand Eric\nlast week, as a starting point to understand the Hermosa LNG\nProject,\nattached are some documents that we have prepared over the past few\nmonths along with a paper from a recent LNG conference on "Offshore\nLNG Unloading."=A0 I would hope this would begin to give you a\nsense of the scope of the project.=A0 There are, of course, many\nissues that need to be addressed.=A0 There has been substantial\nfront-end work performed on the relative feasibility of using\nexisting\noffshore pipeline infrastructure in support of importing LNG into\nCalifornia.=A0 <br>\n<br>\nKey aspects of the Hermosa project are:<br>\n<br>\n</font><font face=3D"Symbol" color=3D"#0000FF">=A8 </font>There is an\nexisting\n(unused) gas pipeline extending from the Hermosa petroleum\nplatform that\ncan be tied into Sempra=92s onshore existing intrastate gas pipeline\ninfrastructure that can carry 400 mmscfd.<br>\n<br>\n<font face=3D"Symbol" color=3D"#0000FF">=A8 </font>There is currently\nadequate\nand idle equipment that could be relocated to provide LNG import\nrates of\n400 mmscfd.<br>\n<br>\n<font face=3D"Symbol" color=3D"#0000FF">=A8 </font>A single ship\nunloading pump\ncan readily and continuously supply LNG at a rate equivalent to 400\nmmscfd.<br>\n<br>\n<font face=3D"Symbol" color=3D"#0000FF">=A8 </font>California is the\nprime\ntarget for increased natural gas supply.<br>\n<br>\n<font face=3D"Symbol" color=3D"#0000FF">=A8 </font>Once importation is\ninitiated through Hermosa, the precedent has been set to construct\na\npermanent, larger floating LNG import terminal<br>\n<br>\nSome of the major areas of concern that are covered with this set\nof\nattachments include:<font face=3D"Symbol" color=3D"#0000FF">\n<dl>\n<dd>=A8 </font>Ability to offload from LNG ship to an existing oil\nplatform.<font face=3D"Symbol" color=3D"#0000FF">\n<dd>=A8 </font>=93Quick Start=94 capability, i.e., timetable when\ncompared to\n=93traditional=94 LNG terminal, assuming such a terminal could be\npermitted.<font face=3D"Symbol" color=3D"#0000FF">\n<dd>=A8 </font>Costs to initiate LNG\nimportation<font face=3D"Symbol" color=3D"#0000FF">\n<dd>=A8 </font>Time-table to initiate LNG\nimportation<font face=3D"Symbol" color=3D"#0000FF">\n<dd>=A8 </font>Long term plans, i.e., how can =93quick start=94 be\ntransitioned\ninto less operator-intensive operation.<br>\n<br>\n\n</dl>Please contact me with any questions you may have about the\nattached\nas it no doubt will not address all questions that might have.=A0\n<br>\n<br>\nBest regards,<br>\n<br>\nJeff Beale<br>\n<br>\n<br>\n<br>\n<font size=3D3\ncolor=3D"#0000FF">=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=\n=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=\n=3D=3D=3D=3D=3D=3D=3D=3D=3D\n=3D=3D<br>\nJeffrey P. Beale, President<br>\nCH=B7IV International <br>\n1120C Benfield Boulevard<br>\nMillersville, MD=A0 USA=A0 21108<br>\nPhone: 410-729-4255<br>\nFax: 410.729.4273<br>\nE-mail: JPBeale@ch-iv.com <br>\n=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=\n=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=\n=3D=3D=3D=3D=3D<br>\n<i>This email and any files transmitted with it from CH=B7IV\nInternational\nor <br>\nCH=B7IV Cryogenics LP are to be considered confidential and\nintended<br>\nsolely for the use of the individual or entity to whom they are\naddressed. <br>\nIf you have received this email in error please notify the\nsender.</font></i></html>\n\n\n\n\n\n\n\n\n\n - Johnston.pdf \n - FRSP-01.pdf \n - 00920-Transfer.pdf \n - 00920-concept.pdf 
    
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
    assert '<html>' not in result
    assert '\x05' not in result
    assert ' --- ' not in result
    assert 'https://www.google.com' not in result
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
    data_loader = LoadEnronData()
    data = data_loader(
        datapath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../resources/enron/sample'
        ),
        try_web=False
    )
    
    assert type(data) == pd.DataFrame

if __name__ == "__main__":
    pytest.main()
