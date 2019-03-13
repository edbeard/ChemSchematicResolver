# -*- coding: utf-8 -*-
"""
test_system
========

Test image processing on images from examples

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

log = logging.getLogger(__name__)

import chemschematicdiagramextractor as csde
import os
import unittest
import copy

from skimage import img_as_float
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


tests_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(os.path.dirname(tests_dir), 'train')
examples_dir = os.path.join(train_dir, 'train_imgs')
markush_dir = os.path.join(train_dir, 'train_markush_small')
labelled_output_dir = os.path.join(train_dir, 'output')

class TestSystem(unittest.TestCase):

    # Testing sementation is sucessful
    def do_segmentation(self, filename, filedir=examples_dir):
        '''
        Tests bounding box assignment for filename

        :param filename:
        :return:
        '''

        test_diag = os.path.join(filedir, filename)

        fig = csde.io.imread(test_diag) # Read in float and raw pixel images
        raw_fig = copy.deepcopy(fig)  # Create unreferenced binary copy

        panels = csde.actions.segment(raw_fig)
        print('Segmented panel number : %s ' % len(panels))
        labels, diags = csde.actions.classify_kmeans(panels)
        labels, diags = csde.actions.preprocessing(labels, diags, fig)
        all_panels = labels + diags
        print('After processing : %s' % len(all_panels))

        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)

        for panel in all_panels:

            diag_rect = mpatches.Rectangle((panel.left, panel.top), panel.width, panel.height,
                                           fill=False, edgecolor='r', linewidth=2)
            ax.text(panel.left, panel.top + panel.height / 4, '[%s]' % panel.tag, size=panel.height / 20, color='r')
            ax.add_patch(diag_rect)

        ax.set_axis_off()
        plt.show()


    def test_segmentation_all(self):

        test_path = examples_dir
        test_imgs = os.listdir(test_path)
        for img_path in test_imgs:
            self.do_segmentation(img_path, filedir=test_path)

    def test_segmentation1(self):

        self.do_segmentation('S014372081630119X_gr1.jpg')

    def test_segmentation2(self):
        self.do_segmentation('S014372081630122X_gr1.jpg')

    def test_segmentation3(self):
        # TODO : noise remover? Get rid of connected components a few pixels in size?
        self.do_segmentation('S014372081630167X_sc1.jpg') # This one isn't identifying the repeating unit, or label

    def test_segmentation4(self):
        self.do_segmentation('S014372081730116X_gr8.jpg')

    def test_segmentation5(self):
        self.do_segmentation('S0143720816300201_sc2.jpg')

    def test_segmentation6(self):
        self.do_segmentation('S0143720816300274_gr1.jpg')

    def test_segmentation7(self):
        self.do_segmentation('S0143720816300419_sc1.jpg')

    def test_segmentation8(self):
        self.do_segmentation('S0143720816300559_sc2.jpg')

    def test_segmentation9(self):
        self.do_segmentation('S0143720816300821_gr2.jpg')

    def test_segmentation10(self):
        self.do_segmentation('S0143720816300900_gr2.jpg')

    def test_segmentation_markush_img(self):
        self.do_segmentation('S0143720816301115_r75.jpg')

    def test_segmentation_markush_img2(self):
        self.do_segmentation('S0143720816300286_gr1.jpg')

    def test_segmentation_markush_img3(self):
        self.do_segmentation('S0143720816301681_gr1.jpg')


    def do_grouping_by_ocr(self):
        '''
        Attempts to identify labels using the ocr module
        :return:
        '''
        pass

            # Testing grouping of diagram - label pairs is correct
    def do_grouping(self, filename, filedir=examples_dir):
        '''
        Tests bounding box assignment for filename

        :param filename:
        :return:
        '''

        test_diag = os.path.join(filedir, filename)

        # Read in float and raw pixel images
        fig = csde.io.imread(test_diag)
        raw_fig = csde.io.imread(test_diag, raw=True)

        # Create unreferenced binary copy
        bin_fig = copy.deepcopy(fig)

        panels = csde.actions.segment(bin_fig)
        labels, diags = csde.actions.classify_kmeans(panels)
        labels, diags = csde.actions.preprocessing(labels, diags, fig)

        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)

       # labelled_diags = csde.actions.classify_kruskal_after_kmeans(labels, diags)
        labelled_diags = csde.actions.label_diags(labels, diags)


        # panels = csde.actions.relabel_panels(panels)
        # diags, labels = csde.actions.classify_kruskal(panels)
        # labelled_diags = csde.actions.label_kruskal(diags, labels)
        #labelled_diags = csde.actions.label_diags(diags, labels)

        colours = iter(['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y','r', 'b', 'g', 'k', 'c', 'm', 'y'])

        for diag in labelled_diags:
            colour = next(colours)

            diag_rect = mpatches.Rectangle((diag.left, diag.top), diag.width, diag.height,
                                      fill=False, edgecolor=colour, linewidth=2)
            ax.add_patch(diag_rect)

            label = diag.label
            label_rect = mpatches.Rectangle((label.left, label.top), label.width, label.height,
                                      fill=False, edgecolor=colour, linewidth=2)
            ax.add_patch(label_rect)

        ax.set_axis_off()
        plt.show()


    def test_grouping_all(self):

        test_path = examples_dir
        test_imgs = os.listdir(test_path)
        for img_path in test_imgs:
            self.do_grouping(img_path, filedir=test_path)


    def test_grouping1(self):
        self.do_grouping('S014372081630119X_gr1.jpg')

    def test_grouping2(self):
        self.do_grouping('S014372081630122X_gr1.jpg')

    def test_grouping3(self):
        self.do_grouping('S014372081630167X_sc1.jpg')

    def test_grouping4(self):
        self.do_grouping('S014372081730116X_gr8.jpg')

    def test_grouping5(self):
        self.do_grouping('S0143720816300201_sc2.jpg')

    def test_grouping6(self):
        self.do_grouping('S0143720816300274_gr1.jpg')

    def test_grouping7(self):
        self.do_grouping('S0143720816300419_sc1.jpg')

    def test_grouping8(self):
        self.do_grouping('S0143720816300559_sc2.jpg')

    def test_grouping9(self):
        self.do_grouping('S0143720816300821_gr2.jpg')

    def test_grouping10(self):
        self.do_grouping('S0143720816300900_gr2.jpg')


    def test_grouping_markush(self):
        self.do_grouping('S0143720816300286_gr1.jpg')


    def do_ocr(self, filename, filedir=examples_dir):
        """ Tests the OCR recognition """


        test_diag = os.path.join(filedir, filename)

        # Read in float and raw pixel images
        fig = csde.io.imread(test_diag)
        raw_fig = csde.io.imread(test_diag, raw=True)

        # Create unreferenced binary copy
        bin_fig = copy.deepcopy(fig)

        panels = csde.actions.segment(bin_fig)
        labels, diags = csde.actions.classify_kmeans(panels)
        labels, diags = csde.actions.preprocessing(labels, diags, bin_fig)

        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)

        labelled_diags = csde.actions.label_diags(diags, labels)


        colours = iter(
            ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y'])

        labels_text = []

        for diag in labelled_diags:
            colour = next(colours)

            diag_rect = mpatches.Rectangle((diag.left, diag.top), diag.width, diag.height,
                                           fill=False, edgecolor=colour, linewidth=2)
            ax.text(diag.left, diag.top + diag.height / 4, '[%s]' % diag.tag, size=diag.height / 20, color='r')
            ax.add_patch(diag_rect)

            label = diag.label
            label_rect = mpatches.Rectangle((label.left, label.top), label.width, label.height,
                                            fill=False, edgecolor=colour, linewidth=2)
            ax.text(label.left, label.top + label.height / 4, '[%s]' % label.tag, size=label.height / 5, color='r')
            ax.add_patch(label_rect)

            label = csde.actions.read_label(fig, label)
            label_strings = [label.text for label in label.text]
            output_label = ' '.join(label_strings)
            labels_text.append(output_label)
            print("Label %s : %s " % (label.tag, labels_text))

        ax.set_axis_off()
        plt.show()

        return labels_text

    def test_ocr_all(self):

        test_path = examples_dir
        test_imgs = os.listdir(test_path)
        for img_path in test_imgs:
            self.do_ocr(img_path, filedir=test_path)

    def test_ocr1(self):
        labels_text = self.do_ocr('S014372081630119X_gr1.jpg')
        labels_text = [text.replace('\n', '') for text in labels_text]
        gold = ['EtNAPH', 'MeNAPH: R=CH3 MeONAPH: R=OCH3']
        for x in gold:
            self.assertIn(x, labels_text)

    # TODO : Update all OCR tests to reflect new format

    def test_ocr2(self):
        labels_text = self.do_ocr('S014372081630122X_gr1.jpg')
        gold = [[['Q1']], [['Q2']], [['Q3']], [['Q4']]]
        for x in gold:
            self.assertIn(x, labels_text)

    def test_ocr3(self):
        labels_text = self.do_ocr('S014372081630167X_sc1.jpg')
        gold = [[['PC71BM']], [['TPE-SQ']]]
        for x in gold:
            self.assertIn(x, labels_text)

    def test_ocr4(self):
        labels_text = self.do_ocr('S014372081730116X_gr8.jpg')
        gold = [[['J51']], [['PDBT-T1']], [['J61']], [['R=2-ethylhexyl'], ['PBDTTT-E-T']], [['R=2-ethylhexyl'], ['PTB7-Th']],
                [['R=2-ethylhexyl'], ['PBDB-T']], [['R=2-ethylhexyl'],['PBDTTT-C-T']]]
        for x in gold:
            self.assertIn(x, labels_text)

    def test_ocr5(self):
        labels_text = self.do_ocr('S0143720816300201_sc2.jpg')
        gold = [[['9', '(>99%)']], [['1','(82%)']], [['3','(86%)']], [['7','(94%)']],
                [['4','(78%)']], [['5','(64%)']], [['2','(78%)']], [['6','(75%)']], [['8','(74%)']]]
        for x in gold:
            self.assertIn(x, labels_text)

    def test_ocr6(self):
        labels = self.do_ocr('S0143720816300274_gr1.jpg')
        gold = [[['8c']], [['8b']], [['8a']], [['7c']], [['7b']], [['7a']]]
        for x in gold:
            self.assertIn(x, labels)

    def test_ocr7(self):
        labels = self.do_ocr('S0143720816300419_sc1.jpg')
        gold = [[['DDOF']], [['DPF']], [['NDOF']], [['PDOF']]]
        for x in gold:
            self.assertIn(x, labels)

    def test_ocr8(self):
        labels = self.do_ocr('S0143720816300559_sc2.jpg')
        gold = [[['1']], [['2']], [['3']]]
        for x in gold:
            self.assertIn(x, labels)

    def test_ocr9(self):
        labels = self.do_ocr('S0143720816300821_gr2.jpg') # Need to add greyscale
        gold = [[['9']], [['10']]]
        for x in gold:
            self.assertIn(x, labels)

    def test_ocr10(self):
        # IR dye doesn't work
        labels = self.do_ocr('S0143720816300900_gr2.jpg')
        gold = [[['ICG']], [['Compound','10']], [['Compound','13']], [['Compound','11']], [['ZW800-1']], [['Compound','12']]]
        for x in gold:
            self.assertIn(x, labels)

    def test_ocr_markush_img(self):
        labels = self.do_ocr('S0143720816301115_r75.jpg')
        gold = [[['PI1-AP']], [['PI1-TMP']], [['PIl-SZ']]]
        for x in gold:
            self.assertIn(x, labels)

    def do_r_group(self, filename, filedir=examples_dir):
        """ Tests the R-group detection and recognition """

        test_diag = os.path.join(filedir, filename)

        # Read in float and raw pixel images
        fig = csde.io.imread(test_diag)
        raw_fig = csde.io.imread(test_diag, raw=True)

        # Create unreferenced binary copy
        bin_fig = copy.deepcopy(fig)

        panels = csde.actions.segment(bin_fig)
        labels, diags = csde.actions.classify_kmeans(panels)
        labels, diags = csde.actions.preprocessing(labels, diags, bin_fig)

        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)

        labelled_diags = csde.actions.label_diags(diags, labels)

        colours = iter(
            ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y'])

        for diag in labelled_diags:
            colour = next(colours)

            diag_rect = mpatches.Rectangle((diag.left, diag.top), diag.width, diag.height,
                                           fill=False, edgecolor=colour, linewidth=2)
            ax.text(diag.left, diag.top + diag.height / 4, '[%s]' % diag.tag, size=diag.height / 20, color='r')
            ax.add_patch(diag_rect)

            label = diag.label
            label_rect = mpatches.Rectangle((label.left, label.top), label.width, label.height,
                                            fill=False, edgecolor=colour, linewidth=2)
            ax.text(label.left, label.top + label.height / 4, '[%s]' % label.tag, size=label.height / 5, color='r')
            ax.add_patch(label_rect)

            diag.label = csde.actions.read_label(fig, label)
            diag = csde.r_group.detect_r_group(diag)

            print(diag.label.r_group)


            # label_strings = [label.text for label in label.text]
            # output_label = ' '.join(label_strings)
            # labels_text.append(output_label)
            # print("Label %s : %s " % (label.tag, labels_text))

        ax.set_axis_off()
        plt.show()

        return labelled_diags

    # def test_ocr_all(self):
    #
    #     test_path = examples_dir
    #     test_imgs = os.listdir(test_path)
    #     for img_path in test_imgs:
    #         self.do_ocr(img_path, filedir=test_path)
    #
    def test_r_group1(self):
        labelled_diags = self.do_r_group('S014372081630119X_gr1.jpg')
        all_detected_r_groups_values = [token[1].text for diag in labelled_diags for tokens in diag.label.r_group for token in tokens]
        gold = ['OCH3', 'CH3']
        for x in gold:
            self.assertIn(x, all_detected_r_groups_values)

        # TODO : Update all OCR tests to reflect new format

    def test_r_group2(self):
        labelled_diags = self.do_r_group('S014372081630122X_gr1.jpg')
        all_detected_r_groups_values = [token[1].text for diag in labelled_diags for tokens in diag.label.r_group for token in tokens]
        self.assertTrue(len(all_detected_r_groups_values) is 0)
    #
    # def test_ocr3(self):
    #     labels_text = self.do_ocr('S014372081630167X_sc1.jpg')
    #     gold = [[['PC71BM']], [['TPE-SQ']]]
    #     for x in gold:
    #         self.assertIn(x, labels_text)
    #

    # TODO : Fix repeated unit logic for this example
    # def test_r_group4(self):
    #     labelled_diags = self.do_r_group('S014372081730116X_gr8.jpg')
    #     all_detected_r_groups_values = [token[1].text for diag in labelled_diags for tokens in diag.label.r_group for token in tokens]
    #     gold = [[['J51']], [['PDBT-T1']], [['J61']], [['R=2-ethylhexyl'], ['PBDTTT-E-T']],
    #             [['R=2-ethylhexyl'], ['PTB7-Th']],
    #             [['R=2-ethylhexyl'], ['PBDB-T']], [['R=2-ethylhexyl'], ['PBDTTT-C-T']]]
    #     for x in gold:
    #         self.assertIn(x, all_detected_r_groups_values)
    #
    def test_r_group5(self):
        labelled_diags = self.do_r_group('S0143720816300201_sc2.jpg')
        all_detected_r_groups_values = [token[1].text for diag in labelled_diags for tokens in diag.label.r_group for token in tokens]
        self.assertTrue(len(all_detected_r_groups_values) is 0)


    def test_r_group6(self):
        labelled_diags = self.do_r_group('S0143720816300274_gr1.jpg')
        all_detected_r_groups_values = [token[1].text for diag in labelled_diags for tokens in diag.label.r_group for token in tokens]
        self.assertTrue(len(all_detected_r_groups_values) is 0)

    def test_r_group7(self):
        labelled_diags = self.do_r_group('S0143720816300419_sc1.jpg')
        all_detected_r_groups_values = [token[1].text for diag in labelled_diags for tokens in diag.label.r_group for token in tokens]
        self.assertTrue(len(all_detected_r_groups_values) is 0)

    def test_r_group8(self):
        labelled_diags = self.do_r_group('S0143720816300559_sc2.jpg')
        all_detected_r_groups_values = [token[1].text for diag in labelled_diags for tokens in diag.label.r_group for token in tokens]
        self.assertTrue(len(all_detected_r_groups_values) is 0)

    def test_r_group9(self):
        labelled_diags = self.do_r_group('S0143720816300821_gr2.jpg')
        all_detected_r_groups_values = [token[1].text for diag in labelled_diags for tokens in diag.label.r_group for token in tokens]
        self.assertTrue(len(all_detected_r_groups_values) is 0)

    def test_r_group10(self):
        labelled_diags = self.do_r_group('S0143720816300900_gr2.jpg')
        all_detected_r_groups_values = [token[1].text for diag in labelled_diags for tokens in diag.label.r_group for token in tokens]
        self.assertTrue(len(all_detected_r_groups_values) is 0)

    def test_r_group11(self):
        labelled_diags = self.do_r_group('S0143720816301115_r75.jpg')
        all_detected_r_groups_values = [token[1].text for diag in labelled_diags for tokens in diag.label.r_group for token in tokens]
        gold = ['S', 'O']
        for x in gold:
            self.assertIn(x, all_detected_r_groups_values)

    def test_r_group12(self):
        labelled_diags = self.do_r_group('S0143720816301681_gr1.jpg')
        all_detected_r_groups_values = [tokens for diag in labelled_diags for tokens in diag.label.r_group]
        unique_combos = []
        for tokens in all_detected_r_groups_values:
            tuple_list = []
            for token in tokens:
                tuple_list.append((token[0].text, token[1].text))
            unique_combos.append(tuple_list)

        gold = [[('X', 'NH'), ('R', 'CN')], [('X', 'NC(O)OEt'), ('R', 'CN')], [('X', 'O'), ('R', 'CN')], [('X', 'O'), ('R', 'H')]]
        for diag in unique_combos:
            self.assertIn(diag, gold)

    def do_r_group_resolution(self, filename, filedir=examples_dir):
        """ Tests the R-group detection, recognition and resoltuion"""


        r_smiles = []
        smiles = []
        test_diag = os.path.join(filedir, filename)

        # Read in float and raw pixel images
        fig = csde.io.imread(test_diag)
        fig_copy = copy.deepcopy(fig) # Unreferenced copy for display
        raw_fig = csde.io.imread(test_diag, raw=True)

        # Create unreferenced binary copy
        bin_fig = copy.deepcopy(fig)
        bin_fig = csde.actions.binarize(bin_fig, threshold=0.7)


        panels = csde.actions.segment(fig)
        labels, diags = csde.actions.classify_kmeans(panels)
        labels, diags = csde.actions.preprocessing(labels, diags, fig)

        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig_copy.img)

        labelled_diags = csde.actions.label_diags(diags, labels)

        colours = iter(
            ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y'])

        for diag in labelled_diags:
            colour = next(colours)

            diag_rect = mpatches.Rectangle((diag.left, diag.top), diag.width, diag.height,
                                           fill=False, edgecolor=colour, linewidth=2)
            ax.text(diag.left, diag.top + diag.height / 4, '[%s]' % diag.tag, size=diag.height / 20, color='r')
            ax.add_patch(diag_rect)

            label = diag.label
            label_rect = mpatches.Rectangle((label.left, label.top), label.width, label.height,
                                            fill=False, edgecolor=colour, linewidth=2)
            ax.text(label.left, label.top + label.height / 4, '[%s]' % label.tag, size=label.height / 5, color='r')
            ax.add_patch(label_rect)

            diag.label = csde.actions.read_label(fig_copy, label)
            diag = csde.r_group.detect_r_group(diag)

            print(diag.label.r_group)


            # label_strings = [label.text for label in label.text]
            # output_label = ' '.join(label_strings)
            # labels_text.append(output_label)
            # print("Label %s : %s " % (label.tag, labels_text))

            if diag.label.r_group != [[]]:
                r_smiles_group = csde.r_group.get_rgroup_smiles(diag, raw_fig)
                for smile in r_smiles_group:
                    r_smiles.append(smile)

            else:

                smile = csde.actions.read_diagram_pyosra(diag, raw_fig)
                smiles.append(smile)

        ax.set_axis_off()
        plt.show()

        total_smiles = r_smiles + smiles

        return total_smiles


    def test_r_group_resolution1(self):
        smiles = self.do_r_group_resolution('S014372081630119X_gr1.jpg')
        print('extracted Smiles are : %s' % smiles)

        gold = ['c1c(ccc(c1)N(c1ccc(/C=C/c2c3c(c(cc2)/C=C/c2ccc(N(c4ccc(C)cc4)c4ccc(cc4)C)cc2)cccc3)cc1)c1ccc(C)cc1)C',
                'c1c(ccc(c1)N(c1ccc(/C=C/c2c3c(c(cc2)/C=C/c2ccc(N(c4ccc(OC)cc4)c4ccc(cc4)OC)cc2)cccc3)cc1)c1ccc(OC)cc1)OC',
                'c1c2n(c3c(c2ccc1)cc(cc3)/C=C/c1ccc(/C=C/c2ccc3n(c4ccccc4c3c2)CC)c2c1cccc2)CC'
                ]

        self.assertEqual(gold, smiles)

    def test_r_group_resolution2(self):
        smiles = self.do_r_group_resolution('S014372081630122X_gr1.jpg')
        print('extracted Smiles are : %s' % smiles)

        # TODO : Try this with tesseract (label resolution is poor)

        gold = [
            ['c1c(ccc(c1)N(c1ccc(/C=C/c2c3c(c(cc2)/C=C/c2ccc(N(c4ccc(C)cc4)c4ccc(cc4)C)cc2)cccc3)cc1)c1ccc(C)cc1)C',
             'c1c(ccc(c1)N(c1ccc(/C=C/c2c3c(c(cc2)/C=C/c2ccc(N(c4ccc(OC)cc4)c4ccc(cc4)OC)cc2)cccc3)cc1)c1ccc(OC)cc1)OC']]

        self.assertEqual(gold, smiles)

    def test_r_group_resolution5(self):
        smiles = self.do_r_group_resolution('S0143720816300201_sc2.jpg')
        print('extracted Smiles are : %s' % smiles)

        gold = ['c1(N(C)C)ccc(cc1)C=C(C#N)C#N', 'n1(c2c(c3c1cccc3)cc(C=C(C#N)C#N)cc2)*',
                                 'c1c2Cc3cc(C=C(C#N)C#N)ccc3c2ccc1', 'o1cccc1C=C(C#N)C#N', 'n1(cccc1C=C(C#N)C#N)C',
                                 's1cccc1C=C(C#N)C#N', 'C[Fe]C1(C*CCC1)C', 'c1ccc(s1)c1sc(cc1)C=C(C#N)C#N']

        self.assertEqual(gold, smiles)


    def test_r_group_resolution6(self):
        smiles = self.do_r_group_resolution('S0143720816300274_gr1.jpg')
        print('extracted Smiles are : %s' % smiles)

        gold = ['CC(c1ccc(c2sc(c3c2[nH]c(n3)c2ccc(c3ccc(/C=C(\\C#N)/C(=O)O)cc3)cc2)c2ccc(C(C)(C)C)cc2)cc1)(C)C',
                                'CC(c1ccc(c2sc(c3nc([nH]c23)c2ccc(c3sc(/C=C(\\C#N)/C(=O)O)cc3)cc2)c2ccc(C(C)(C)C)cc2)cc1)(C)C',
                                'CC(c1ccc(c2sc(c3c2[nH]c(n3)c2sc(cc2)c2ccc(/C=C(\\C#N)/C(=O)O)s2)c2ccc(C(C)(C)C)cc2)cc1)(C)C',
                                'c1(cccs1)c1sc(c2nc(n(c12)CCCC)c1ccc(c2ccc(/C=C(\\C#N)/C(=O)O)cc2)cc1)c1sccc1',
                                'c1cc(sc1)c1sc(c2c1n(CCCC)c(n2)c1ccc(c2ccc(/C=C(\\C#N)/C(=O)O)s2)cc1)c1sccc1',
                                'c1(cccs1)c1sc(c2nc(n(c12)CCCC)c1sc(cc1)c1sc(cc1)/C=C(\\C#N)/C(=O)O)c1sccc1']

        self.assertEqual(gold, smiles)

    def test_r_group_resolution7(self):
        smiles = self.do_r_group_resolution('S0143720816300286_gr1.jpg')
        print('extracted Smiles are : %s' % smiles)

        gold = ['c1(N(C)C)ccc(cc1)C=C(C#N)C#N', 'n1(c2c(c3c1cccc3)cc(C=C(C#N)C#N)cc2)*',
                'c1c2Cc3cc(C=C(C#N)C#N)ccc3c2ccc1', 'o1cccc1C=C(C#N)C#N', 'n1(cccc1C=C(C#N)C#N)C',
                's1cccc1C=C(C#N)C#N', 'C[Fe]C1(C*CCC1)C', 'c1ccc(s1)c1sc(cc1)C=C(C#N)C#N']

        self.assertEqual(gold, smiles)

    def test_r_group_resolution8(self):
        smiles = self.do_r_group_resolution('S0143720816300419_sc1.jpg')
        print('extracted Smiles are : %s' % smiles)

        gold = ['c12c3ccc(cc3C(=O)c2cc(cc1)c1ccc(C(=O)C)cc1)c1ccc(C(=O)C)cc1', 'c12c3c(C(=O)c2cc(c2ccccc2)cc1)cc(c1ccccc1)cc3',
                'c12c3c(C(=O)c2cc(cc1)c1ccc(C(=O)OC)cc1)cc(c1ccc(C(=O)OC)cc1)cc3',
                'c12c3c(C(=O)c2cc(cc1)c1ccc(C=O)cc1)cc(c1ccc(C=O)cc1)cc3']

        self.assertEqual(gold, smiles)

    def test_r_group_resolution9(self):
        smiles = self.do_r_group_resolution('S0143720816300559_sc2.jpg')
        print('extracted Smiles are : %s' % smiles)

        gold = ['c1ccc2nc(oc2c1)c1c(O)c(O)c(c2oc3c(n2)cccc3)s1','c1ccc2c(c1)nc(o2)c1c(O)c(OCC)c(c2oc3c(n2)cccc3)s1',
                'c1ccc2c(c1)nc(o2)c1c(OCC)c(OCC)c(c2oc3c(n2)cccc3)s1']

        self.assertEqual(gold, smiles)

    def test_r_group_resolution10(self):
        smiles = self.do_r_group_resolution('S0143720816300821_gr2.jpg')
        print('extracted Smiles are : %s' % smiles)

        gold = ['c1c2C(=O)c3ccc(c(Nc4cc(c(N)c5C(=O)c6c(C(=O)c45)cccc6)C)c3C(=O)c2ccc1)C',
                'c1(ccccc1)Nc1c2C(=O)c3ccccc3C(=O)c2c(Nc2cc(c(N)c3C(=O)c4c(C(=O)c23)cccc4)C)c(C)c1']

        self.assertEqual(gold, smiles)

    def test_r_group_resolution11(self):
        smiles = self.do_r_group_resolution('S0143720816300900_gr2.jpg')

        # TODO : Currently broken. Likely due to difficul-to-parse + and - signs in circles
        print('extracted Smiles are : %s' % smiles)

        gold = ['c1c2C(=O)c3ccc(c(Nc4cc(c(N)c5C(=O)c6c(C(=O)c45)cccc6)C)c3C(=O)c2ccc1)C',
                'c1(ccccc1)Nc1c2C(=O)c3ccccc3C(=O)c2c(Nc2cc(c(N)c3C(=O)c4c(C(=O)c23)cccc4)C)c(C)c1']

        self.assertEqual(gold, smiles)

    def test_r_group_resolution12(self):
        # TODO : This diagram still fails in resolving the : in the label of R-group diagram
        smiles = self.do_r_group_resolution('S0143720816301115_r75.jpg')
        print('extracted Smiles are : %s' % smiles)

        gold = ['c1c2C(=O)c3ccc(c(Nc4cc(c(N)c5C(=O)c6c(C(=O)c45)cccc6)C)c3C(=O)c2ccc1)C',
                'c1(ccccc1)Nc1c2C(=O)c3ccccc3C(=O)c2c(Nc2cc(c(N)c3C(=O)c4c(C(=O)c23)cccc4)C)c(C)c1']

        self.assertEqual(gold, smiles)

    def test_r_group_resolution13(self):
        smiles = self.do_r_group_resolution('S0143720816301681_gr1.jpg')
        print('extracted Smiles are : %s' % smiles)

        gold = ['c1c2C(=O)c3ccc(c(Nc4cc(c(N)c5C(=O)c6c(C(=O)c45)cccc6)C)c3C(=O)c2ccc1)C',
                'c1(ccccc1)Nc1c2C(=O)c3ccccc3C(=O)c2c(Nc2cc(c(N)c3C(=O)c4c(C(=O)c23)cccc4)C)c(C)c1']

        self.assertEqual(gold, smiles)

    def test_r_group_resolution_var_var(self):
        """ test a document case where variables are set equal to each other"""

        smiles = self.do_r_group_resolution('S0143720816301115_gr1.jpg')
        print('extracted Smiles are : %s' % smiles)

        gold = ['c1c2C(=O)c3ccc(c(Nc4cc(c(N)c5C(=O)c6c(C(=O)c45)cccc6)C)c3C(=O)c2ccc1)C',
                'c1(ccccc1)Nc1c2C(=O)c3ccccc3C(=O)c2c(Nc2cc(c(N)c3C(=O)c4c(C(=O)c23)cccc4)C)c(C)c1']

       # self.assertEqual(gold, smiles)



    def do_osra(self, filename):
        """ Tests the OSRA chemical diagram recognition """

        test_diag = os.path.join(examples_dir, filename)

        # Read in float and raw pixel images
        fig = csde.io.imread(test_diag)
        raw_fig = csde.io.imread(test_diag, raw=True)

        # Create unreferenced binary copy
        bin_fig = copy.deepcopy(fig)

        panels = csde.actions.segment(bin_fig)
        panels = csde.actions.preprocessing(panels, bin_fig)

        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)

        diags, labels = csde.actions.classify_kruskal(panels)
        labelled_diags = csde.actions.label_kruskal(diags, labels)

        colours = iter(
            ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm',
             'y'])

        smiles = []

        for diag in labelled_diags:
            colour = next(colours)

            diag_rect = mpatches.Rectangle((diag.left, diag.top), diag.width, diag.height,
                                           fill=False, edgecolor=colour, linewidth=2)
            ax.text(diag.left, diag.top + diag.height / 4, '[%s]' % diag.tag, size=diag.height / 20, color='r')
            ax.add_patch(diag_rect)

            label = diag.label
            label_rect = mpatches.Rectangle((label.left, label.top), label.width, label.height,
                                            fill=False, edgecolor=colour, linewidth=2)
            ax.text(label.left, label.top + label.height / 4, '[%s]' % label.tag, size=label.height / 5, color='r')
            ax.add_patch(label_rect)

            smile, confidence = csde.actions.read_diagram(fig, diag)
            if '*' not in smile:
                print(smile, confidence)
            smiles.append(smile)
            print("Label {} ({}): {} ".format(diag.tag, confidence, smile))


        ax.set_axis_off()
        plt.savefig(os.path.join(labelled_output_dir, filename))

        return smiles

    def test_osra2(self):
        smiles = self.do_osra('S014372081630122X_gr1.jpg')
        print(smiles)

        """ Output with .jpg on 300dpi
        Label 4 (7.9463): N#C/C(=C\c1ccc(s1)c1ccc(c2c1*=C(*)C(=*2)**)c1ccc(s1)c1ccc(cc1)*(c1ccc(cc1)*)c1ccc(cc1)*)/C(=O)O 
Label 1 (5.9891): N#C/C(=C\c1ccc(s1)c1ccc(c2c1nc(**)c(n2)*)c1ccc(s1)c1ccc(cc1)*(c1ccccc1)c1ccccc1)/C(=O)O 
Label 5 (6.1918): N#C/C(=C\c1ccc(s1)c1cc(*)c(cc1*)c1ccc(s1)c1ccc(cc1)N(c1ccccc1)c1ccccc1)/C(=O)O 
Label 0 (9.4724): CCCCCCc1nc2c(c3ccc(s3)/C=C(/C(=O)O)\C#N)c3*=C(**)C(=*c3c(c2nc1*)c1ccc(s1)c1ccc(cc1)N(c1ccccc1)c1ccccc1)** """

    def test_osra3(self):
        smiles = self.do_osra('S014372081630167X_sc1.jpg')
        print(smiles)

    def test_osra5(self):
        smiles = self.do_osra('S0143720816300201_sc2.jpg')
        print(smiles)


    def test_osra6(self):
        smiles = self.do_osra('S0143720816300274_gr1.jpg')
        print(smiles)

    def test_osra7(self):
        smiles = self.do_osra('S0143720816300419_sc1.jpg')
        print(smiles)

    def test_osra8(self):
        smiles = self.do_osra('S0143720816300559_sc2.jpg')
        print(smiles)

    def test_osra9(self):
        smiles = self.do_osra('S0143720816300821_gr2.jpg')
        print(smiles)

    def test_osra10(self):
        # IR dye doesn't work
        smiles= self.do_osra('S0143720816300900_gr2.jpg')
        print(smiles)


class TestValidation(unittest.TestCase):

    def do_metrics(self, filename):
        """ Used to identify correlations between metrics and output validity"""

        test_fig = os.path.join(examples_dir, filename)

        # Read in float and raw pixel images
        fig = csde.io.imread(test_fig)
        raw_fig = csde.io.imread(test_fig, raw=True)

        # Create unreferenced binary copy
        bin_fig = copy.deepcopy(fig)

        panels = csde.actions.segment(bin_fig)
        panels = csde.actions.preprocessing(panels, fig)

        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)

        diags, labels = csde.actions.classify_kruskal(panels)
        labelled_diags = csde.actions.label_kruskal(diags, labels)

        colours = iter(
            ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm',
             'y'])

        smiles = []

        avg_pixel_ratio = csde.validate.total_pixel_ratio(bin_fig, labelled_diags)
        diags_to_image_ratio = csde.validate.diagram_to_image_area_ratio(bin_fig, labelled_diags)
        avg_diag_area_to_total_img_ratio = csde.validate.avg_diagram_area_to_image_area(bin_fig, labelled_diags)

        for diag in labelled_diags:
            colour = next(colours)

            diag_rect = mpatches.Rectangle((diag.left, diag.top), diag.width, diag.height,
                                           fill=False, edgecolor=colour, linewidth=2)
            ax.text(diag.left, diag.top + diag.height / 4, '[%s]' % diag.tag, size=diag.height / 20, color='r')
            ax.add_patch(diag_rect)

            label = diag.label
            label_rect = mpatches.Rectangle((label.left, label.top), label.width, label.height,
                                            fill=False, edgecolor=colour, linewidth=2)
            ax.text(label.left, label.top + label.height / 4, '[%s]' % label.tag, size=label.height / 5, color='r')
            ax.add_patch(label_rect)

            smile, confidence = csde.actions.read_diagram(fig, diag)
            smiles.append(smile)
            print("Label {} ({}): {} ".format(diag.tag, confidence, smile))
            print("Black pixel ratio : %s " % csde.validate.pixel_ratio(bin_fig, diag))

        print('Overall diagram metrics:')
        print('Average 1 / all ratio: %s' % avg_pixel_ratio)
        print('Average diag / fig area ratio: %s' % avg_diag_area_to_total_img_ratio)
        print('Diag number to fig area ratio: %s' % diags_to_image_ratio)


        ax.set_axis_off()
        plt.savefig(os.path.join(labelled_output_dir, filename))

        return smiles

    def test_validation2(self):
        smiles = self.do_metrics('S014372081630122X_gr1.jpg')

    def test_validation3(self):
        smiles = self.do_metrics('S014372081630167X_sc1.jpg')

    def test_validation5(self):
        smiles = self.do_metrics('S0143720816300201_sc2.jpg')

    def test_validation6(self):
        smiles = self.do_metrics('S0143720816300274_gr1.jpg')

    def test_validation7(self):
        smiles = self.do_metrics('S0143720816300419_sc1.jpg')

    def test_validation8(self):
        smiles = self.do_metrics('S0143720816300559_sc2.jpg')

    def test_validation9(self):
        smiles = self.do_metrics('S0143720816300821_gr2.jpg')

    def test_validation10(self):
        # IR dye doesn't work
        smiles= self.do_metrics('S0143720816300900_gr2.jpg')


class TestFiltering(unittest.TestCase):
    """ Tests the results filtering via wildcard removal and pybel validation """

    def do_filtering(self, filename):
        """ Used to identify correlations between metrics and output validity"""
        test_fig = os.path.join(examples_dir, filename)

        # Read in float and raw pixel images
        fig = csde.io.imread(test_fig)
        raw_fig = csde.io.imread(test_fig, raw=True)

        # Create unreferenced binary copy
        bin_fig = copy.deepcopy(fig)

        # Preprocessing steps
        panels = csde.actions.segment(bin_fig)
        panels = csde.actions.preprocessing(panels, fig)

        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)

        # Get label pairs
        diags, labels = csde.actions.classify_kruskal(panels)
        labelled_diags = csde.actions.label_kruskal(diags, labels)

        colours = iter(
            ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm',
                'y'])

        diags_with_smiles = []

        for diag in labelled_diags:
            colour = next(colours)

            diag_rect = mpatches.Rectangle((diag.left, diag.top), diag.width, diag.height,
                                            fill=False, edgecolor=colour, linewidth=2)
            ax.text(diag.left, diag.top + diag.height / 4, '[%s]' % diag.tag, size=diag.height / 20, color='r')
            ax.add_patch(diag_rect)

            label = diag.label
            label_rect = mpatches.Rectangle((label.left, label.top), label.width, label.height,
                                            fill=False, edgecolor=colour, linewidth=2)
            ax.text(label.left, label.top + label.height / 4, '[%s]' % label.tag, size=label.height / 5, color='r')
            ax.add_patch(label_rect)

            smile, confidence = csde.actions.read_diagram(fig, diag)
            diag.smile = smile
            diags_with_smiles.append(diag)

        # Run post-processing: 
        formatted_smiles = csde.validate.format_all_smiles(diags_with_smiles)
        print(formatted_smiles)
        return formatted_smiles

    def test_filtering2(self):
        smiles = self.do_filtering('S014372081630122X_gr1.jpg')

    def test_filtering3(self):
        smiles = self.do_filtering('S014372081630167X_sc1.jpg')

    def test_filtering5(self):
        smiles = self.do_filtering('S0143720816300201_sc2.jpg')

    def test_filtering6(self):
        smiles = self.do_filtering('S0143720816300274_gr1.jpg')

    def test_filtering7(self):
        smiles = self.do_filtering('S0143720816300419_sc1.jpg')

    def test_filtering8(self):
        smiles = self.do_filtering('S0143720816300559_sc2.jpg')

    def test_filtering9(self):
        smiles = self.do_filtering('S0143720816300821_gr2.jpg')

    def test_filtering10(self):
        # IR dye doesn't work
        smiles= self.do_filtering('S0143720816300900_gr2.jpg')