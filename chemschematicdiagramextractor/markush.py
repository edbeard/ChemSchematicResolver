# -*- coding: utf-8 -*-
"""
Scripts for identifying Markush structures

========
author: Ed Beard
email: ejb207@cam.ac.uk

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

log = logging.getLogger(__name__)

class MarkushInterpretter():
    ''' Identifies and interprets Markush interpretters'''

    def __init__(self, img):
        self.img = img

    def find_markush_candidates(self):
        ''' Finds potential regions to test for Markush
            Based on the precese of the "=" delimiter'''







    def interpret_text(self):
        ''' Use NLP (inc. CDE?) to parse the extracted text
            will return specifier (eg. R) and chemical formula (eg. CO2)'''

    def match_structure(self):
        ''' This might not be here - could be implemented from '''

