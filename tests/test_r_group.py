# -*- coding: utf-8 -*-
"""
test_parse
========

Test R-group resolution operations.

"""

from chemschematicdiagramextractor import r_group
#from molvs import standardize_smiles
import unittest

class TestRgroup(unittest.TestCase):
    """ Test functios from the r_group.py module"""

    def do_resolve(self, comp):
        raw_smile = r_group.resolve_structure(comp)
        #std_smile = standardize_smiles(raw_smile)
        return raw_smile


    def test_resolve_structure_1(self):

        comp = '4-nitrophenyl'
        gold = '[O-][N+](=O)c1ccccc1'
        result = self.do_resolve(comp)
        self.assertEqual(gold, result)


    def test_resolve_structure_2(self):

        comp = '2-chloro-4-nitrophenol'
        gold = '[O-][N+](=O)c1ccccc1'
        result = self.do_resolve(comp)
        self.assertEqual(gold, result)


    def test_resolve_structure_3(self):

        comp = '5-nitrothiazol-2-yl'
        gold = '[O-][N+](=O)c1ccccc1'
        result = self.do_resolve(comp)
        self.assertEqual(gold, result)


