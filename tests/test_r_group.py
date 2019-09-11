# -*- coding: utf-8 -*-
"""
test_parse
========

Test R-group resolution operations.

"""

from chemschematicresolver import r_group

import unittest


def do_resolve(comp):
    raw_smile = r_group.resolve_structure(comp)
    return raw_smile


class TestRgroup(unittest.TestCase):
    """ Test functios from the r_group.py module"""

    def test_resolve_structure_1(self):

        comp = '4-nitrophenyl'
        gold = '[O-][N+](=O)c1ccccc1'
        result = do_resolve(comp)
        self.assertEqual(gold, result)

    def test_resolve_structure_2(self):

        comp = '2-chloro-4-nitrophenol'
        gold = 'Oc1ccc(cc1Cl)[N+]([O-])=O'
        result = do_resolve(comp)
        self.assertEqual(gold, result)

    def test_resolve_structure_4(self):

        comp = 'Hexyl'
        gold = '[O-][N+](=O)c1cc(c(Nc2c(cc(cc2[N+]([O-])=O)[N+]([O-])=O)[N+]([O-])=O)c(c1)[N+]([O-])=O)[N+]([O-])=O'
        result = do_resolve(comp)
        self.assertEqual(gold, result)
