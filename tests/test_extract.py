# -*- coding: utf-8 -*-
"""
test_extract
========

Test extraction of Chemical Schematic Images

"""

import unittest
import os
import chemschematicdiagramextractor as csde
from skimage.transform import rescale
import copy

tests_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(os.path.dirname(tests_dir), 'train')
train_markush_small_dir = os.path.join(train_dir, 'train_markush_small')
r_group_diag_dir = os.path.join(train_dir, 'r_group_diags')


class TestExtract(unittest.TestCase):
    """ Tests the overall extraction case"""

    def do_extract_small(self, filename, gold):
        """ Extract images from the small markush directory"""

        path = os.path.join(train_markush_small_dir, filename)
        result = csde.extract.extract_diagram(path, debug=False)
        self.assertEqual(gold, result)

        return result

    def test_r_group_extract(self):

        gold = [(['EtNAPH'], 'c1c2n(c3c(c2ccc1)cc(cc3)/C=C/c1ccc(/C=C/c2ccc3n(c4ccccc4c3c2)CC)c2c1cccc2)CC'),
                (['MeNAPH'], 'c1c(ccc(c1)N(c1ccc(/C=C/c2c3c(c(cc2)/C=C/c2ccc(N(c4ccc(C)cc4)c4ccc(cc4)C)cc2)cccc3)cc1)c1ccc(C)cc1)C'),
                (['MeONAPH'], 'c1c(ccc(c1)N(c1ccc(/C=C/c2c3c(c(cc2)/C=C/c2ccc(N(c4ccc(OC)cc4)c4ccc(cc4)OC)cc2)cccc3)cc1)c1ccc(OC)cc1)OC')]

        self.do_extract_small('S014372081630119X_gr1.jpg', gold)

    def test_r_group_extract2(self):

        # This example fails as the chemical names need to be resolved

        gold = []

        self.do_extract_small('S0143720816300286_gr1.jpg', gold)

    def test_r_group_extract3(self):

        # This one fails because it can't resolve the : symbol correcty.
        # This means the R-group variable is wrong and that no label is extracted
        # Can be identified as a FP from * in smile, and lack of label

        gold = []

        self.do_extract_small('S0143720816301115_r75.jpg', gold)

    def test_r_group_extract4(self):

        # Identified one of the numeric labels as an atom indicator (inside osra)
        # Could be solved by doing a numerical clean

        gold = []

        self.do_extract_small('S0143720816301681_gr1.jpg', gold)

    def do_extract_r_group_diags(self, filename, gold):
        """ Extract images from the small markush directory"""

        path = os.path.join(r_group_diag_dir, filename)
        result = csde.extract.extract_diagram(path, debug=False)
        self.assertEqual(gold, result)

    def test_r_group_diag_1(self):

        # Identified numerical points as labels

        gold = []

        self.do_extract_r_group_diags('S0143720816301565_gr1.jpg', gold)

    def test_r_group_diag_2(self):

        # Extraction of the R-group for diagram B - need to add logic for the 'or' identification
        # Also need t check that Octyl and Hexyl can be converted into the Rgroup - might need to try using CIRPY

        gold = []

        self.do_extract_r_group_diags('S0143720816302054_sc1.jpg', gold)

    def test_r_group_diag_3(self):

        # Has assigned the same label to both diagrams, resulting in different outcomes
        # Either some of the 4 R-values are not resolved, or something else is going wrong on the rgroup diag

        gold = []

        self.do_extract_r_group_diags('S0143720816302108_gr1.jpg', gold)
