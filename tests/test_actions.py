# -*- coding: utf-8 -*-
"""
test_actions
========

Test image processing actions.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

log = logging.getLogger(__name__)

import chemschematicresolver as csr
import os
from pathlib import Path
import unittest
import copy
import numpy

from skimage import img_as_float
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


tests_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(os.path.dirname(tests_dir), 'train')
markush_dir = os.path.join(train_dir, 'train_markush_small')
sample_diag = os.path.join(markush_dir, 'S014372081630119X_gr1.jpg')

class TestActions(unittest.TestCase):

    def test_binarization(self):
        ''' Tests binarization of image'''

        fig = csr.io.imread(sample_diag)
        bin = csr.actions.binarize(fig)
        self.assertTrue(True in bin.img and False in bin.img)

    def test_segement(self):
        ''' Tests segmentation of image'''

        fig = csr.io.imread(sample_diag)
        raw_fig = csr.io.imread(sample_diag, raw=True)  # Reads in version of pure pixels

        bin_fig = copy.deepcopy(fig)  # Image copy to be binarized

        float_fig = copy.deepcopy(fig)  # Image copy to be converted to float
       # float_fig.img = img_as_float(float_fig.img)

        #bin_fig = csr.actions.binarize(bin_fig) # Might not need binary version?
        #bin_fig.img = img_as_float(bin_fig.img)
        panels = csr.actions.segment(bin_fig)

        # Create debugging image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)
        #train_dir = os.path.join(os.path.dirname(tests_dir), 'train')

        diags, labels = csr.actions.classify(panels)
        #
        for panel in diags:
            rect = mpatches.Rectangle((panel.left, panel.top), panel.width, panel.height,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(panel.left, panel.top + panel.height / 4, '[%s]' % panel.tag, size=panel.height / 20, color='r')

        for panel in labels:
            rect = mpatches.Rectangle((panel.left, panel.top), panel.width, panel.height,
                                      fill=False, edgecolor='yellow', linewidth=2)
            ax.text(panel.left, panel.top + panel.height / 4, '[%s]' % panel.tag, size=panel.height / 5, color='r')
            ax.add_patch(rect)

        ax.set_axis_off()
        plt.show()
        # for diag in diags:
        #     csr.actions.assign_label_to_diag(diag, labels)
        labelled_diags = csr.actions.label_diags(diags, labels)
        tagged_diags = csr.actions.read_all_labels(fig, labelled_diags)
        tagged_resolved_diags = csr.actions.read_all_diags(raw_fig, tagged_diags)

    def test_kruskal(self):

        p1 = csr.model.Panel(-1, 1, -1, 1, 0)
        p2 = csr.model.Panel(2, 4, 3, 5, 1)
        p3 = csr.model.Panel(6, 8, 23, 25, 2)

        panels = [p1, p2, p3]

        sorted_edges = csr.actions.kruskal(panels)
        print(sorted_edges)
        self.assertEqual(sorted_edges[0][2], 5.)
        self.assertEqual(round(sorted_edges[1][2]), 20)

    def test_merge_rect(self):

        r1 = csr.model.Rect(0, 10, 0, 20)
        r2 = csr.model.Rect(5, 15, 5, 15)

        merged_r = csr.actions.merge_rect(r1, r2)
        self.assertEqual(merged_r.left, 0)
        self.assertEqual(merged_r.right, 15)
        self.assertEqual(merged_r.top, 0)
        self.assertEqual(merged_r.bottom, 20)

    def test_horizontal_merging(self):
        ''' Tests the horizontal merging is behaving'''

        test_markush = os.path.join(markush_dir, 'S0143720816301681_gr1.jpg')
        fig = csr.io.imread(test_markush)
        raw_fig = copy.deepcopy(fig)  # Create unreferenced binary copy

        panels = csr.actions.segment(raw_fig)
        print('Segmented panel number : %s ' % len(panels))

        # Crete output image (post-merging)
        merged_panels = csr.actions.merge_label_horizontally_repeats(panels)

        out_fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.imshow(fig.img)

        for panel in merged_panels:
            diag_rect = mpatches.Rectangle((panel.left, panel.top), panel.width, panel.height,
                                           fill=False, edgecolor='r', linewidth=2)
            ax2.add_patch(diag_rect)

        plt.show()
