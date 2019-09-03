# -*- coding: utf-8 -*-
"""
eval_markush
========

Used to test accuracy of training samples in semi-automatic way

"""

import unittest
import os
import copy
import chemschematicresolver as csr
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from chemschematicresolver.ocr import LABEL_WHITELIST


tests_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(os.path.dirname(tests_dir), 'train')
raw_train_data = os.path.join(train_dir, 'train_markush_small')


class TestMarkush(unittest.TestCase):
    '''
    Test the stages in the Markush (R-Group) detection and resolution pipeline
    '''

    def find_labels_from_img(self, filename):

        train_img = os.path.join(raw_train_data, filename)
        # Read in float and raw pixel images
        fig = csr.io.imread(train_img)
        raw_fig = csr.io.imread(train_img, raw=True)

        # Create unreferenced binary copy
        bin_fig = copy.deepcopy(fig)

        panels = csr.actions.segment(bin_fig)
        panels = csr.actions.preprocessing(panels, fig)

        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)

        diags, labels = csr.actions.classify_kruskal(panels)
        labelled_diags = csr.actions.label_kruskal(diags, labels)

        colours = iter(
            ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm',
             'y'])

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

        ax.set_axis_off()
        plt.show()


        return fig, labelled_diags

    def test_markush_candidate_detection(self):

        fig, labelled_diags = self.find_labels_from_img('S0143720816300286_gr1.jpg')
        test_diag = labelled_diags[1]
        print(csr.actions.read_label(fig, test_diag.label))

        print(labelled_diags)

    def test_general_ocr(self):

        train_img = os.path.join(raw_train_data, 'S0143720816301681_gr1.jpg')
        # Read in float and raw pixel images
        fig = csr.io.imread(train_img)
        txt = csr.ocr.get_text(fig.img, whitelist=LABEL_WHITELIST)
        print(txt)



