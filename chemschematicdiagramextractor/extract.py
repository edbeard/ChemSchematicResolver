# -*- coding: utf-8 -*-
"""
Functions for extracting Diagrams

========

Toolkit for extracting diagram-label pairs from schematic chemical diagrams.

"""

from .io import imread
from .actions import segment, classify_kmeans, preprocessing, label_diags, read_label, read_diagram_pyosra, clean_output
from .r_group import detect_r_group, get_rgroup_smiles
from .validate import is_false_positive

import copy
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def extract_diagram(filename, debug=False):
    """ Converts a chemical diagram to SMILES string and extracted label candidates

    :return : List of label candidates and smiles
    :rtype : list[tuple[list[string],string]]
    """

    # Output lists
    r_smiles = []
    smiles = []

    # Read in float and raw pixel images
    fig = imread(filename)

    # Create unreferenced binary copy
    bin_fig = copy.deepcopy(fig)

    # Create image of raw pixels
    raw_fig = imread(filename, raw=True)

    # Segment image into pixel islands
    panels = segment(bin_fig)

    # Classify panels into labels and diagrams by size
    labels, diags = classify_kmeans(panels)

    # Proprocess image (eg merge labels that are small into larger labels)
    labels, diags = preprocessing(labels, diags, bin_fig)

    if debug is True:
        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)
        colours = iter(
            ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y'])

    # Add label information to the appropriate diagram by expanding bounding box
    labelled_diags = label_diags(diags, labels)

    for diag in labelled_diags:

        label = diag.label

        if debug is True:

            colour = next(colours)

            # Add diag bbox to debug image
            diag_rect = mpatches.Rectangle((diag.left, diag.top), diag.width, diag.height,
                                           fill=False, edgecolor=colour, linewidth=2)
            ax.text(diag.left, diag.top + diag.height / 4, '[%s]' % diag.tag, size=diag.height / 20, color='r')
            ax.add_patch(diag_rect)

            # Add label bbox to debug image
            label_rect = mpatches.Rectangle((label.left, label.top), label.width, label.height,
                                            fill=False, edgecolor=colour, linewidth=2)
            ax.text(label.left, label.top + label.height / 4, '[%s]' % label.tag, size=label.height / 5, color='r')
            ax.add_patch(label_rect)

        # Read the label
        # TODO : Fix logic so it's a method of the label class
        diag.label = read_label(fig, label)

        # Add r-group variables if detected
        diag = detect_r_group(diag)

        # Get SMILES for output
        smiles, r_smiles = get_smiles(diag, raw_fig, smiles, r_smiles)

    print( " The results are :")
    print('R-smiles %s' % r_smiles)
    print('Smiles %s' % smiles)
    if debug is True:
        ax.set_axis_off()
        plt.show()

    total_smiles = smiles + r_smiles

    # Removing false positives from lack of labels or wildcard smiles
    output = [smile for smile in total_smiles if is_false_positive(smile) is False]

    return output


def get_smiles(diag, fig, smiles, r_smiles):
    """ Identifies diagrams containing R-group"""

    # Resolve R-groups if detected
    if len(diag.label.r_group) > 0:
        r_smiles_group = get_rgroup_smiles(diag, fig)
        for smile in r_smiles_group:
            label_cand_str = [cand.text for cand in smile[0]]
            r_smiles.append((label_cand_str, smile[1]))

    # Resolve diagram normally if no R-groups - should just be one smile
    else:
        smile = read_diagram_pyosra(diag, fig)
        label_raw = diag.label.text
        label_cand_str = [clean_output(cand.text) for cand in label_raw]

        smiles.append((label_cand_str, smile))

    return smiles, r_smiles
