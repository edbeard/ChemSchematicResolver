# -*- coding: utf-8 -*-
"""
eval_osra
========

Used to test accuracy of training samples in semi-automatic way

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
from matplotlib import pyplot as plt
import csv

# Paths used in training:

tests_dir = os.path.dirname(os.path.abspath(__file__))
raw_train_data = os.path.join(os.path.dirname(tests_dir), 'examples', 'train_imgs')
seg_train_dir = os.path.join(tests_dir, 'train_osra_small')
seg_train_csv = os.path.join(tests_dir, 'train_osra.csv')


def split_raw_train_data():
    """ Splits the raw training data into separate images"""

    for train_fig in os.listdir(raw_train_data):

        train_path = os.path.join(raw_train_data, train_fig)
        #  Read in float and raw pixel images
        fig = csde.io.imread(train_path)
        raw_fig = csde.io.imread(train_path, raw=True)

        # Create unreferenced binary copy
        bin_fig = copy.deepcopy(fig)

        # Segment images
        panels = csde.actions.segment(bin_fig)
        panels = csde.actions.preprocessing(panels, fig)

        # Classify diagrams and their labels using kruskal
        diags, labels = csde.actions.classify_kruskal(panels)
        labelled_diags = csde.actions.label_kruskal(diags, labels)

        # Save the segmented diagrams
        for diag in labelled_diags:
            # Save the segmented diagrams
            l, r, t, b = diag.left, diag.right, diag.top, diag.bottom
            cropped_img = csde.actions.crop(raw_fig.img, l, r, t, b)
            out_path = os.path.join(seg_train_dir, train_fig[:-4] + '_' + str(diag.tag))
            csde.io.imsave(out_path + '.png', cropped_img)

            # TODO :  Create image with regions and labels superimposed
            



# class TestOsra(unittest.TestCase):
#     """ Class tests whether output of OSRA is correct, with human input.
#     Tests pass if > 80% of results after filtering are correct.
#     """
#
#     def test_train_data(self):
#         """ Looks in the training data to get smiles """
#
#         tps, fps = [], [] # Define true and false positive counters
#
#         # Create file if it doesn't exist
#         if not os.path.isfile(seg_train_csv):
#             open(seg_train_csv, 'w')
#
#         with open(seg_train_csv, "r") as f:
#             csv_reader = csv.reader(f)
#             auto_results = list(csv_reader)
#             tp_prev = [res for res in auto_results if res[2] is 'y']
#             fp_prev = [res for res in auto_results if res[2] is 'n']
#
#         for train_fig in os.listdir(seg_train_dir):
#
#
#             # if train_fig in tp_filenames:
#             #     tps.append([train_fig, auto_results[tp_filenames.index(train_fig)], 'y'])
#             # else:
#
#             img_path = os.path.join(seg_train_dir, train_fig)
#
#             # Load in cropped diagram
#             fig = csde.io.imread(img_path)
#             l, r, t, b = csde.actions.get_img_boundaries(fig.img)
#             diag = csde.model.Diagram(l, r, t, b, 0) # Using throwaway tag 0
#
#             # Get the SMILES and the confidence
#             smile, confidence = csde.actions.read_diagram(fig, diag)
#
#             if '*' in smile:
#                 pass # Remove wildcard results
#             elif [train_fig, smile, 'y'] in tp_prev:
#                 tps.append([train_fig, smile, 'y'])
#             elif [train_fig, smile, 'n'] in fp_prev:
#                 fps.append([train_fig, smile, 'n'])
#             else:
#                 while True:
#                     inp = str(input('Filename : %s , smile: %s ; Correct? [y/n]\n' % (train_fig, smile)))
#                     if inp.lower() in ['y', '']:
#                         tps.append([train_fig, smile, 'y'])
#                         break
#                     elif inp.lower() in ['n']:
#                         fps.append([train_fig, smile, 'n'])
#                         break
#                     else :
#                         print("Invalid response. Please try again.")
#
#         print("Precision : %s" % str(float(len(tps))/ (float(len(tps)) + float(len(fps)))))
#
#         with open(seg_train_csv, "w") as f:
#             csv_writer = csv.writer(f)
#             csv_writer.writerows(tps)
#             csv_writer.writerows(fps)

def eval_train_data():
    """ Looks in the training data to get smiles """

    tps, fps = [], [] # Define true and false positive counters

    # Create file if it doesn't exist
    if not os.path.isfile(seg_train_csv):
        open(seg_train_csv, 'w')

    with open(seg_train_csv, "r") as f:
        csv_reader = csv.reader(f)
        auto_results = list(csv_reader)
        tp_prev = [res for res in auto_results if res[2] is 'y']
        fp_prev = [res for res in auto_results if res[2] is 'n']

    for train_fig in os.listdir(seg_train_dir):


        # if train_fig in tp_filenames:
        #     tps.append([train_fig, auto_results[tp_filenames.index(train_fig)], 'y'])
        # else:

        img_path = os.path.join(seg_train_dir, train_fig)

        # Load in cropped diagram
        fig = csde.io.imread(img_path)
        l, r, t, b = csde.actions.get_img_boundaries(fig.img)
        diag = csde.model.Diagram(l, r, t, b, 0) # Using throwaway tag 0

        # Get the SMILES and the confidence
        smile, confidence = csde.actions.read_diagram(fig, diag)

        if '*' in smile:
            pass # Remove wildcard results
        elif [train_fig, smile, 'y'] in tp_prev:
            tps.append([train_fig, smile, 'y'])
        elif [train_fig, smile, 'n'] in fp_prev:
            fps.append([train_fig, smile, 'n'])
        else:
            while True:
                inp = str(input('Filename : %s , smile: %s ; Correct? [y/n]\n' % (train_fig, smile)))
                if inp.lower() in ['y', '']:
                    tps.append([train_fig, smile, 'y'])
                    break
                elif inp.lower() in ['n']:
                    fps.append([train_fig, smile, 'n'])
                    break
                else :
                    print("Invalid response. Please try again.")

    print("Precision : %s" % str(float(len(tps))/ (float(len(tps)) + float(len(fps)))))

    with open(seg_train_csv, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(tps)
        csv_writer.writerows(fps)


                

if __name__ == '__main__':
    eval_train_data()

