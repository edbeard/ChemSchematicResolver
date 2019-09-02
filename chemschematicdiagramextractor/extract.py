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
import os
import urllib
import cirpy
#from molvs import standardize_smiles

from chemdataextractor import Document
from chemdataextractor.text.normalize import chem_normalize


def extract_document(filename, do_extract=True, output=os.path.join(os.path.dirname(os.getcwd()), 'csd')):
    """ Extracts chemical records from a document and identifies chemcial schematic diagrams.
    Then substitutes in if the label was found in a record

    :param filename: Location of document to be extracted
    :param do_extract : Boolean indicating whether images should be extracted
    :param output: Directory to store extracted images

    : return : Dictionary of chemical records
    """

    # Extract the raw records from CDE
    doc = Document.from_file(filename)

    figs = doc.figures

    # Identify image candidates
    csds = find_image_candidates(figs, filename)

    # Donwload figures locally
    fig_paths = download_figs(csds, output)
    print("All relevant figures from %s downloaded sucessfully" % filename)

    if do_extract:
        # Run CSDE
        results = []
        for path in fig_paths:
            try:
                results.append(extract_diagram(path))
            except:
                pass
        # Subsitute smiles for labels

        subsitute_labels(doc.records.serialize(), results)

        return results


def subsitute_labels(records, results):
    """ Looks for label candidates in the document records and subsitutes where appropriate"""

    # TODO : make it so this substitutes in the CDE records with new field smiles: ['diagram': ''] or something...

    doc_named_records = []

    record_labels = [record for record in records if 'labels' in record.keys()]

    # Get all records that contain common labels
    for diag_result in results:
        for label_cands, smile in diag_result:
            for record_label in record_labels:
                overlap = [(record_label, label_cand, smile)  for label_cand in label_cands if label_cand in record_label['labels']]
                doc_named_records += overlap

    print(doc_named_records)
    #
    for doc_record, diag_label, diag_smile in doc_named_records:
        for name in (doc_record['names']):
            try:
                doc_smile = cirpy.resolve(chem_normalize(name).encode('utf-8'), 'smiles')
                # doc_smile = standardize_smiles(doc_smile)
                # diag_smile = standardize_smiles(diag_smile, 'smiles')
            except:
                pass
            print('Doc smile: %s ' % doc_smile)
            print('Diag smile: %s \n' % diag_smile)


def download_figs(figs, output):
    """ Downloads figures from url

    :param figs: List of tuples in form figure metadat (Filename, figure id, url to figure, caption)
    :param output: Location of output images
    """

    if not os.path.exists(output):
        os.makedirs(output)

    fig_paths = []

    for file, id, url, caption in figs:

        img_format = url.split('.')[-1]
        print('Downloading %s image from %s' % (img_format, url))
        filename = file.split('/')[-1].rsplit('.', 1)[0] + '_' + id + '.' + img_format
        path = os.path.join(output, filename)

        print("Downloading %s..." % filename)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path) # Saves downloaded image to file
        else:
            print("File exists! Going to next image")

        fig_paths.append(path)

    return fig_paths


def find_image_candidates(figs, filename):
    """ Returns a list of csd figures

    :param figs: ChemDataExtractor figure objects
    :return: List of figure metadata (Filename, figure id, url to figure, caption)
    :rtype:   list[tuple[string, string, string, string]]
    """
    csd_imgs = []

    for fig in figs:
        detected = False  # Used to avoid processing images twice
        records = fig.records
        caption = fig.caption
        for record in records:
            if detected is True:
                break

            rec = record.serialize()
            if ['csd'] in rec.values():
                detected = True
                print('Chemical schematic diagram instance found!')
                csd_imgs.append((filename, fig.id, fig.url, caption.text.replace('\n', ' ')))

    return csd_imgs


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
    fig_bbox = fig.get_bounding_box()

    # Create unreferenced binary copy
    bin_fig = copy.deepcopy(fig)

    # Create image of raw pixels
    raw_fig = imread(filename, raw=True)

    # Segment image into pixel islands
    panels = segment(bin_fig)

    # Classify and preprocess images, to account for merging in segmentation
    labels, diags = classify_kmeans(panels, fig)

    # Preprocess image (eg merge labels that are small into larger labels)
    labels, diags = preprocessing(labels, diags, fig)

    # Re-cluster by height if there are more Diagram objects than Labels
    if len(labels) < len(diags):
        labels, diags = classify_kmeans(panels, fig, skel=False)
        labels, diags = preprocessing(labels, diags, fig)

        #TODO: Add some logic here to choose the clustering that has the closet number of diagrams and labels?

    if debug is True:
        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)
        colours = iter(
            ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y'])

    # Add label information to the appropriate diagram by expanding bounding box
    labelled_diags = label_diags(labels, diags, fig_bbox)

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
        smiles, r_smiles = get_smiles(diag, smiles, r_smiles)

    print("The results are :")
    print('R-smiles %s' % r_smiles)
    print('Smiles %s' % smiles)
    if debug is True:
        ax.set_axis_off()
        plt.show()

    total_smiles = smiles + r_smiles

    # Removing false positives from lack of labels or wildcard smiles
    output = [smile for smile in total_smiles if is_false_positive(smile) is False]
    print('Final Results : ')
    for result in output:
        print(result)

    return output


def get_smiles(diag, smiles, r_smiles):
    """ Identifies diagrams containing R-group"""

    # Resolve R-groups if detected
    if len(diag.label.r_group) > 0:
        r_smiles_group = get_rgroup_smiles(diag)
        for smile in r_smiles_group:
            label_cand_str = [cand.text for cand in smile[0]]
            r_smiles.append((label_cand_str, smile[1]))

    # Resolve diagram normally if no R-groups - should just be one smile
    else:
        smile = read_diagram_pyosra(diag)
        label_raw = diag.label.text
        label_cand_str = [clean_output(cand.text) for cand in label_raw]

        smiles.append((label_cand_str, smile))

    return smiles, r_smiles
