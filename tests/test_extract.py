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
train_imgs_dir = os.path.join(train_dir, 'train_imgs')


class TestExtract(unittest.TestCase):
    """ Tests the overall extraction case"""

    def do_extract_small(self, filename, gold):
        """ Extract images from the small markush directory"""

        path = os.path.join(train_markush_small_dir, filename)
        result = csde.extract.extract_diagram(path, debug=True)
        self.assertEqual(gold, result)

        return result

    def do_extract_all_imgs(self, dir_name=train_imgs_dir):
        """ Run extraction on the 'train_imgs' directory (no assertions)"""

        test_path = train_imgs_dir
        test_imgs = os.listdir(test_path)
        for img_path in test_imgs:
            full_path = os.path.join(test_path, img_path)
            csde.extract.extract_diagram(full_path, debug=True)

    def test_run_train_imgs(self):
        """ Run all images in train_imgs directory"""
        self.do_extract_all_imgs()

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

        # NB : This test is automatically filtering out the failed R-group resolution from the presence of wildcards

        gold = [(['107'], 'c1c2ccc3cccc4c3c2c(cc4)cc1/C=N/CCSSCC/N=C/c1cc2ccc3cccc4ccc(c1)c2c34'),
                (['106'], 'c1c2ccc3cc(/C=N/c4ccccc4O)c(O)c4c3c2c(cc1)cc4')]

        self.do_extract_small('S0143720816301115_r75.jpg', gold)

    def test_r_group_extract4(self):

        # Identified one of the numeric labels as an atom indicator (inside osra)
        # Could be solved by doing a numerical clean

        gold = [(['1'], 'N(CC)(CC)c1ccc2cc(C#N)c(=N)oc2c1'),
                (['2'], 'N(CC)(CC)c1ccc2cc(C#N)/c(=N/C(=O)OCC)/oc2c1'),
                (['3'], 'N(CC)(CC)c1ccc2cc(C#N)c(=O)oc2c1'),
                (['4'], 'N(CC)(CC)c1ccc2ccc(=O)oc2c1')]

        self.do_extract_small('S0143720816301681_gr1.jpg', gold)

    def do_extract_r_group_diags(self, filename, gold):
        """ Extract images from the small markush directory"""

        path = os.path.join(r_group_diag_dir, filename)
        result = csde.extract.extract_diagram(path, debug=True)
        self.assertEqual(gold, result)

    def test_r_group_diag_1(self):

        # Identified numerical points as labels

        gold = []

        self.do_extract_r_group_diags('S0143720816301565_gr1.jpg', gold)

    def test_r_group_diag_2(self):

        # Octyl and Hectyl resolved correctly. NB the smiles are still wrong though due to OSRA failing
        # Need to improve the relative label-detection as well : at the moment it's just returning all candidates

        gold = [(['A'],
                'c1(ccc(s1)c1cc(F)c(c2cc3S4(c5c(c3s2)sc(c5)c2c([F]CC(C4)CCCC)cc(c3sc(c4ccc(CCCCCC)s4)cc3)c3nsnc23)CC(CC)CCCC)c2nsnc12)c1sc(CCCCCC)cc1'),
                (['1'],
                'N1(C(=O)c2c(C1=O)ccc(c1sc(c3cc(F)c(c4cc5S6(c7c(c5s4)sc(c7)c4c([F]CC(C6)CCCC)cc(c5sc(cc5)c5ccc6C(=O)N(C(=O)c6c5)CCCCCCCC)c5nsnc45)CC(CC)CCCC)c4nsnc34)cc1)c2)CCCCCCCC'),
                (['2'],
                'N1(C(=O)c2c(C1=O)ccc(c1sc(c3cc(F)c(c4cc5S6(c7c(c5s4)sc(c7)c4c([F]CC(C6)CCCC)cc(c5sc(cc5)c5ccc6C(=O)N(C(=O)c6c5)CCCCCC)c5nsnc45)CC(CC)CCCC)c4nsnc34)cc1)c2)CCCCCC')]

        self.do_extract_r_group_diags('S0143720816302054_sc1.jpg', gold)

    def test_r_group_diag_3(self):

        # Has assigned the same label to both diagrams, resulting in different outcomes
        # Either some of the 4 R-values are not resolved, or something else is going wrong on the rgroup diag

        gold = []

        self.do_extract_r_group_diags('S0143720816302108_gr1.jpg', gold)

    def test_r_group_diag_4(self):

        # Schematic flow diagram, including arrows etc

        gold = []

        self.do_extract_r_group_diags('S0143720816301826_sc1.jpg', gold)


    def test_r_group_diag_5(self):

        # Containing segmenting lines, and large R-groups

        gold = []

        self.do_extract_r_group_diags('S0143720816301401_gr5.jpg', gold)

    def do_extract_train_imgs(self, filename, gold):
        """ Extract images from the train_imgs directory"""

        path = os.path.join(train_imgs_dir, filename)
        result = csde.extract.extract_diagram(path, debug=False)
        self.assertEqual(gold, result)

    def test_train_imgs_1(self):

        gold = []

        self.do_extract_train_imgs('S0143720816301115_gr1.jpg', gold)
