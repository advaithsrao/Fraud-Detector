import os
import sys
sys.path.append("..")

import pandas as pd
import pytest

from detector.preprocessor import Preprocessor
from utils.util_preprocessor import convert_string_to_list, add_subject_to_body


@pytest.fixture
def mail():
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
    
    \n>\n>\n> Best,
    Joe Smith
    
    Joe Smith | Strategy & Business Development
    111 Market St. Suite 111| San Francisco, CA 94103
    M: 111.111.1111| joe@foobar.com
    """

@pytest.fixture
def subject_body():
    return {
        'subject': 'Urgent Reply Needed',
        'body': 'Hello, your Boss here. Quickly send me a gift card, so I can buy some stuff for the customer.'
    }

@pytest.fixture
def users_string():
    return """
        Lay, Kenneth &&
        Skilling, Jeff &&
        Howard, Kevin &&
        Krautz, Michael &&
        Yeager, Scott
    """

def test_subject_body(subject_body):
    assert add_subject_to_body(subject_body['subject'],subject_body['body']) == 'Urgent Reply Needed Hello, your Boss here. Quickly send me a gift card, so I can buy some stuff for the customer.'

def test_convert_string_to_list(users_string):
    assert convert_string_to_list(users_string, sep='&&') == ['Lay, Kenneth', 'Skilling, Jeff', 'Howard, Kevin', 'Krautz, Michael', 'Yeager, Scott']

def test_preprocesor(mail):
    preprocess = Preprocessor()
    result = preprocess(mail)
    
    #remove urls
    assert 'https://www.google.com' not in result
    
    #remove html tags
    assert '<html>' not in result
    assert '<head>' not in result
    assert '<body>' not in result
    assert '<p>' not in result
    assert '<href' not in result
    assert '<font' not in result
    assert '<i>' not in result
    assert '<b>' not in result
    assert '<br>' not in result
    assert '<dl>' not in result

    #remove new lines
    assert '\n' not in result

    #remove unicode
    assert '\x05' not in result

    #remove specific patterns
    assert '-+Original Message-+' not in result
    assert 'From:' not in result
    assert 'Sent:' not in result
    assert 'To:' not in result
    assert 'Cc:' not in result
    assert 'Subject:' not in result

    #remove non-alphanumeric tokens
    assert ' --- ' not in result

    #remove multiple whitespace
    assert '  ' not in result

    #remove signature
    assert 'Joe Smith' not in result
    assert 'Strategy & Business Development' not in result
    assert 'M: 111.111.1111| joe@foobar.com' not in result

if __name__ == "__main__":
    pytest.main()
