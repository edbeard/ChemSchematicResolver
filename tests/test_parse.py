# -*- coding: utf-8 -*-
"""
test_parse
========

Test parsing operations.

"""

from chemdataextractor.parse.cem import BaseParser, lenient_chemical_label
from chemdataextractor.doc.text import Sentence
from chemschematicresolver.parse import LabelParser

import unittest

class TestParse(unittest.TestCase):
    ''' Checks that the chemical label extraction logic is working'''

    def test_label_parsing(self):

        test_sentence = Sentence('3', parsers=[LabelParser()])
        self.assertEqual(test_sentence.records.serialize(), [{'labels': ['3']}])
