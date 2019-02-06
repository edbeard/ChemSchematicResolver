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

import chemschematicdiagramextractor as csde
import os
from pathlib import Path
import unittest
import copy
import numpy

from skimage import img_as_float
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


tests_dir = os.path.dirname(os.path.abspath(__file__))
markush_dir = os.path.join(os.path.dirname(tests_dir), 'tests', 'train_markush_small')
sample_diag = os.path.join(markush_dir, 'S014372081630119X_gr1.jpg')

class TestActions(unittest.TestCase):

    def test_binarization(self):
        ''' Tests binarization of image'''

        fig = csde.io.imread(sample_diag)
        bin = csde.actions.binarize(fig)
        self.assertTrue(True in bin.img and False in bin.img)

    def test_segement(self):
        ''' Tests segmentation of image'''

        fig = csde.io.imread(sample_diag)
        raw_fig = csde.io.imread(sample_diag, raw=True)  # Reads in version of pure pixels

        bin_fig = copy.deepcopy(fig)  # Image copy to be binarized

        float_fig = copy.deepcopy(fig)  # Image copy to be converted to float
       # float_fig.img = img_as_float(float_fig.img)

        #bin_fig = csde.actions.binarize(bin_fig) # Might not need binary version?
        #bin_fig.img = img_as_float(bin_fig.img)
        tag_img, panels = csde.actions.segment(bin_fig)

        # Create debugging image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)
        #
        diags, labels = csde.actions.classify(panels)
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
        #     csde.actions.assign_label_to_diag(diag, labels)
        labelled_diags = csde.actions.label_diags(diags, labels)
        tagged_diags = csde.actions.read_all_labels(fig, labelled_diags)
        tagged_resolved_diags = csde.actions.read_all_diags(raw_fig, tagged_diags)

    def test_kruskal(self):

        p1 = csde.model.Panel(-1, 1, -1, 1, 0)
        p2 = csde.model.Panel(2, 4, 3, 5, 1)
        p3 = csde.model.Panel(6, 8, 23, 25, 2)

        panels = [p1, p2, p3]

        sorted_edges = csde.actions.kruskal(panels)
        print(sorted_edges)
        self.assertEqual(sorted_edges[0][2], 5.)
        self.assertEqual(round(sorted_edges[1][2]), 20)

    def test_merge_rect(self):

        r1 = csde.model.Rect(0, 10, 0, 20)
        r2 = csde.model.Rect(5, 15, 5, 15)

        merged_r = csde.actions.merge_rect(r1, r2)
        self.assertEqual(merged_r.left, 0)
        self.assertEqual(merged_r.right, 15)
        self.assertEqual(merged_r.top, 0)
        self.assertEqual(merged_r.bottom, 20)
