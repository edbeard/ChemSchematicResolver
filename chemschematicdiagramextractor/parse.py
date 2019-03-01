# -*- coding: utf-8 -*-
"""
Classes for parsing relevant info

========
author: Ed Beard
email: ejb207@cam.ac.uk

"""

from chemdataextractor.parse.cem import BaseParser, lenient_chemical_label
from chemdataextractor.utils import first
from chemdataextractor.model import Compound
from lxml import etree

class LabelParser(BaseParser):

    root = lenient_chemical_label

    def interpret(self, result, start, end):
        for label in result.xpath('./text()'):
            yield Compound(labels=[label])