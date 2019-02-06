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
import numpy as np

from skimage import img_as_float
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


tests_dir = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(tests_dir, 'train_imgs')
labelled_output_dir = os.path.join(tests_dir, 'output')

class TestSystem(unittest.TestCase):

    # Testing sementation is sucessful
    def do_segmentation(self, filename):
        '''
        Tests bounding box assignment for filename

        :param filename:
        :return:
        '''

        test_diag = os.path.join(example_dir, filename)

        fig = csde.io.imread(test_diag) # Read in float and raw pixel images
        raw_fig = copy.deepcopy(fig)  # Create unreferenced binary copy

        panels = csde.actions.segment(raw_fig)
        print('Segmented panel number : %s ' % len(panels))
        panels = csde.actions.preprocessing(panels, fig)
        print('After processing : %s' % len(panels))

        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)

        for panel in panels:

            diag_rect = mpatches.Rectangle((panel.left, panel.top), panel.width, panel.height,
                                           fill=False, edgecolor='r', linewidth=2)
            ax.text(panel.left, panel.top + panel.height / 4, '[%s]' % panel.tag, size=panel.height / 20, color='r')
            ax.add_patch(diag_rect)

        ax.set_axis_off()
        plt.show()

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

    def test_segmentation11(self):
        self.do_segmentation('S0143720816301097_gr1.jpg')

    def test_segmentation_markush_img(self):
        self.do_segmentation('S0143720816301115_r75.jpg')

            # Testing grouping of diagram - label pairs is correct
    def do_grouping(self, filename):
        '''
        Tests bounding box assignment for filename

        :param filename:
        :return:
        '''

        test_diag = os.path.join(example_dir, filename)

        # Read in float and raw pixel images
        fig = csde.io.imread(test_diag)
        raw_fig = csde.io.imread(test_diag, raw=True)

        # Create unreferenced binary copy
        bin_fig = copy.deepcopy(fig)

        panels = csde.actions.segment(bin_fig)
        panels = csde.actions.preprocessing(panels, fig)

        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)

        diags, labels = csde.actions.classify_kruskal(panels)
        labelled_diags = csde.actions.label_kruskal(diags, labels)
        #labelled_diags = csde.actions.label_diags(diags, labels)

        colours = iter(['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y','r', 'b', 'g', 'k', 'c', 'm', 'y'])

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

        # for panel in diags:
        #     rect = mpatches.Rectangle((panel.left, panel.top), panel.width, panel.height,
        #                               fill=False, edgecolor='red', linewidth=2)
        #
        #
        # for panel in labels:
        #     rect = mpatches.Rectangle((panel.left, panel.top), panel.width, panel.height,
        #                               fill=False, edgecolor='yellow', linewidth=2)
        #     ax.text(panel.left, panel.top + panel.height / 4, '[%s]' % panel.tag, size=panel.height, color='r')
        #     ax.add_patch(rect)

        ax.set_axis_off()
        plt.show()
        # for diag in diags:
        #     csde.actions.assign_label_to_diag(diag, labels)
        # labelled_diags = csde.actions.label_diags(diags, labels)
        # tagged_diags = csde.actions.read_all_labels(fig, labelled_diags)
        # tagged_resolved_diags = csde.actions.read_all_diags(raw_fig, tagged_diags)

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

    def test_grouping11(self):
        self.do_grouping('S0143720816301097_gr1.jpg')

    def test_grouping_markush(self):
        self.do_grouping('S0143720816300286_gr1.jpg')


    def do_ocr(self, filename):
        """ Tests the OCR recognition """


        test_diag = os.path.join(example_dir, filename)

        # Read in float and raw pixel images
        fig = csde.io.imread(test_diag)
        raw_fig = csde.io.imread(test_diag, raw=True)

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

            label_text = csde.actions.read_label(fig, label)
            labels_text.append(label_text)
            print("Label %s : %s " % (label.tag, label_text))

        ax.set_axis_off()
        plt.show()

        return labels_text

    def test_ocr1(self):
        labels_text = self.do_ocr('S014372081630119X_gr1.jpg')
        gold = [[['EtNAPH']], [['MeNAPH:', 'R=CH3'], ['MeONAPH:', 'R=OCH3']]]
        for x in gold:
            self.assertIn(x, labels_text)

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

    def test_ocr11(self):
        # currently failing for 3 reasons :
        # 1 - the sentences are being split up for PI1
        # 2 - Z's are being identified as 2's
        # 3 - one of the '1s' is identfied as 'l'
        labels = self.do_ocr('S0143720816301097_gr1.jpg')
        gold = [[['PI1-AP']], [['PI1-TMP']], [['PIl-SZ']]]
        for x in gold:
            self.assertIn(x, labels)

    def test_ocr_markush_img(self):
        labels = self.do_ocr('S0143720816301115_r75.jpg')
        gold = [[['PI1-AP']], [['PI1-TMP']], [['PIl-SZ']]]
        for x in gold:
            self.assertIn(x, labels)

    def do_osra(self, filename):
        """ Tests the OSRA chemical diagram recognition """

        test_diag = os.path.join(example_dir, filename)

        # Read in float and raw pixel images
        fig = csde.io.imread(test_diag)
        raw_fig = csde.io.imread(test_diag, raw=True)

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

    def test_osra11(self):
        # currently failing for 3 reasons :
        # 1 - the sentences are being split up for PI1
        # 2 - Z's are being identified as 2's
        # 3 - one of the '1s' is identfied as 'l'
        labels = self.do_ocr('S0143720816301097_gr1.jpg')
        gold = [[['PI1-AP']], [['PI1-TMP']], [['PIl-SZ']]]
        for x in gold:
            self.assertIn(x, labels)


class TestValidation(unittest.TestCase):

    def do_metrics(self, filename):
        """ Used to identify correlations between metrics and output validity"""

        test_fig = os.path.join(example_dir, filename)

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
        test_fig = os.path.join(example_dir, filename)

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