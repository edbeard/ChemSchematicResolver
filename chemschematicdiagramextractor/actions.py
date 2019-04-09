# -*- coding: utf-8 -*-
"""
Image processing actions

========

A toolkit of image processing actions for segmentation

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import numpy as np
from skimage.color import rgb2gray
from skimage import morphology
from skimage.util import pad, crop
from skimage.util import crop as crop_skimage
from skimage.morphology import binary_closing, disk
from skimage.measure import regionprops
import subprocess
import os
import itertools
import copy
from chemdataextractor.doc.text import Sentence

from scipy import ndimage as ndi
from sklearn.cluster import KMeans
# TODO : rename after removing cmd line read diagram logic
import osra_rgroup

from .model import Panel, Diagram, Label, Rect, Graph, Figure
from .ocr import get_text, get_sentences, get_words, PSM, LABEL_WHITELIST
from .io import imsave, imdel
from .parse import LabelParser

log = logging.getLogger(__name__)


def crop(img, left=None, right=None, top=None, bottom=None):
    """Crop image.

    Automatically limits the crop if bounds are outside the image.

    :param numpy.ndarray img: Input image.
    :param int left: Left crop.
    :param int right: Right crop.
    :param int top: Top crop.
    :param int bottom: Bottom crop.
    :return: Cropped image.
    :rtype: numpy.ndarray
    """
    height, width = img.shape[:2]

    left = max(0, 0 if left is None else left )
    right = min(width, width if right is None else right)
    top = max(0, 0 if top is None else top)
    bottom = min(height, height if bottom is None else bottom)
    out_img = img[top: bottom, left : right ]
    return out_img


def binarize(fig, threshold=0.85):
    """ Converts image to binary

    RGB images are converted to greyscale using :class:`skimage.color.rgb2gray` before binarizing.

    :param numpy.ndarray img: Input image
    :param float|numpy.ndarray threshold: Threshold to use.
    :return: Binary image.
    :rtype: numpy.ndarray
    """

    img = fig.img

    # Skip if already binary
    if img.ndim <= 2 and img.dtype == bool:
        return img

    img = convert_greyscale(img)

    # TODO: Investigate Niblack and Sauvola threshold methods
    # Binarize with threshold (default of 0.9 empirically determined)
    binary = img < threshold
    fig.img = binary
    return fig


def binary_close(fig, size=20):
    """ Joins unconnected pixel by dilation and erosion"""
    selem = disk(size)

    fig.img = pad(fig.img, size, mode='constant')
    fig.img = binary_closing(fig.img, selem)
    fig.img = crop_skimage(fig.img, size)
    return fig


def binary_floodfill(fig):
    """ Converts all pixels inside closed contour to 1"""
    fig.img = ndi.binary_fill_holes(fig.img)
    return fig


def binary_tag(fig):
    """ Tag connected regions with pixel value of 1"""
    fig.img, no_tagged = ndi.label(fig.img)
    return fig, no_tagged


def convert_greyscale(img):
    """ Converts to greyscale if RGB"""

    # Convert to greyscale if needed
    if img.ndim == 3 and img.shape[-1] in [3, 4]:
        rgb_img = copy.deepcopy(img)
        grey_img = rgb2gray(img)
    else:
        grey_img = img
    return grey_img


def get_bounding_box(fig):
    """ Gets the bounding box of each segment"""
    panels =[]
    regions = regionprops(fig.img)
    for region in regions:
        y1,x1,y2,x2 = region.bbox
        panels.append(Panel(x1,x2,y1,y2, region.label - 1)) # Sets tags to start from 0
    return panels


def get_repeating_unit(labels, diags, fig):
    """ Identifies 'n' labels as repeating unit identifiers"""

    # TODO : Alter this logic to be done at normal 'read label' time?
    # Could prevent reading of labels twice unecessarily...

    ns = []

    for diag in diags:
        for cand in labels:
            if diag.overlaps(cand):

                repeating_units = [token for sentence in read_label(fig, cand).text for token in sentence.tokens if 'n' is token.text]
                if repeating_units:
                    ns.append(cand)
                    diag.repeating = True

    labels = [label for label in labels if label not in ns]
    return labels, diags


def relabel_panels(panels):
    """ Relabel panels"""

    for i, panel in enumerate(panels):
        panel.tag = i
    return panels


def segment(fig, size=20):
    """ Segments image """

    bin_fig = binarize(fig)

    closed_fig = binary_close(bin_fig, size)

    fill_img = binary_floodfill(closed_fig)
    tag_img, no_tagged = binary_tag(fill_img)
    panels = get_bounding_box(tag_img)
    return panels


def classify_kmeans(panels):
    """Takes the input images, then classifies through k means cluster of the panel area"""

    if len(panels) <= 1:
        raise Exception('Only one panel detected. Cannot cluster')
    return get_labels_and_diagrams_k_means_clustering(panels)


def preprocessing(labels, diags, fig):
    """Preprocessing steps, expand as needed"""

    # Remove repeating unit indicators
    labels, diags = get_repeating_unit(labels, diags, fig)

    diags = remove_diag_pixel_islands(diags, fig)

    label_candidates_horizontally_merged = merge_label_horizontally(labels)
    label_candidates_fully_merged = merge_labels_vertically(label_candidates_horizontally_merged)
    labels_converted = convert_panels_to_labels(label_candidates_fully_merged)
    return labels_converted, diags


def get_diagram_numbers(diags, fig):
    """ Removes vertex numbers from diagrmas for cleaner OSRA resolution"""

    num_bbox = []
    for diag in diags:

        diag_text = read_diag_text(fig, diag)

        # Simplify into list comprehension when working...
        for token in diag_text:
            if token.text in '123456789':
                print("Numeral sucessfully extracted %s" % token.text)
                num_bbox.append((diag.left + token.left, diag.left + token.right,
                                 diag.top + token.top, diag.top + token.bottom))

    # Make a cleaned copy of image to be used when resolving diagrams
    diag_fig = copy.deepcopy(fig)

    for bbox in num_bbox:
        diag_fig.img[bbox[2]:bbox[3], bbox[0]:bbox[1]] = np.ones(3)

    return diag_fig


def remove_diag_pixel_islands(diags, fig):
    """ Removes all small pixel islands from the diagram """

    for diag in diags:

        # Make a cleaned copy of image to be used when resolving diagrams
        clean_fig = copy.deepcopy(fig)

        diag_fig = Figure(crop(clean_fig.img, diag.left, diag.right, diag.top, diag.bottom))
        seg_img = Figure(crop(clean_fig.img, diag.left, diag.right, diag.top, diag.bottom))
        sub_panels = segment(seg_img, size=13)

        panel_areas = [panel.area for panel in sub_panels]
        diag_area = max(panel_areas)

        sub_panels = [panel for panel in sub_panels if panel.area != diag_area]

        sub_bbox = [(panel.left, panel.right, panel.top, panel.bottom) for panel in sub_panels]

        for bbox in sub_bbox:
            diag_fig.img[bbox[2]:bbox[3], bbox[0]:bbox[1]] = np.ones(3)

        diag.fig = diag_fig

    return diags



def remove_diagram_numbers(nums_bboxes, fig):
    """ Removes floating numbers from the diagram regions of image """

    for bbox in nums_bboxes:
        fig[bbox[2]:bbox[3], bbox[0], bbox[1]] = np.ones(3)

    # Remove after debugging:
    # import matplotlib.pyplot as plt
    # out_fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(fig.img)
    # plt.show()




def classify(panels):
    """ Classifies diagrams and labels for panels  """

    diags = []
    labels = []

    areas = [panel.area for panel in panels]
    mean = np.mean(areas)# Threshold for classification
    thresh = mean - mean*0.5

    for panel in panels:
        if panel.area > thresh:
            diags.append(Diagram(panel.left, panel.right, panel.top, panel.bottom, panel.tag))
        else:
            labels.append(Label(panel.left, panel.right, panel.top, panel.bottom, panel.tag))

    return diags, labels


def kruskal(panels):
    """ Runs Kruskals algorithm on input Panel objects"""

    g = Graph(len(panels))
    for a, b in itertools.combinations(panels, 2):
        g.addEdge(a.tag, b.tag, a.separation(b))

    sorted_edges = g.kruskal()
    return sorted_edges


def assign_labels_diagram_pairs_after_kmeans(labels, diagrams):
    """Assigns label-digaram pairs after doing kmeans to identify"""


def classify_kruskal_after_kmeans(labels, diagrams):
    """Pairs up diagrams and pairs after clustering via kmeans"""

    ordered_diags = []
    ordered_labels = []
    used_tags = []

    converted_labels = [Label(label.left, label.right, label.top, label.bottom, 0) for label in labels]
    converted_diagrams = [Diagram(diag.left, diag.right, diag.top, diag.bottom, 0) for diag in diagrams]
    sorted_labels_and_diags = relabel_panels(converted_diagrams + converted_labels)

    sorted_edges = kruskal(sorted_labels_and_diags)

    for edge in sorted_edges:

        for panel in sorted_labels_and_diags:
            if panel.tag == edge[0]:
                p1 = panel
            elif panel.tag == edge[1]:
                p2 = panel

        if p1.tag in used_tags or p2.tag in used_tags:
            pass
        elif type(p1).__name__ == 'Diagram' and type(p2).__name__ == 'Label':
            ordered_diags.append(p1)
            ordered_labels.append(p2)
            used_tags.extend([p1.tag, p2.tag])

        elif type(p2).__name__ == 'Diagram' and type(p1).__name__ == 'Label':
            ordered_diags.append(p2)
            ordered_labels.append(p1)
            used_tags.extend([p1.tag, p2.tag])

    print('Pairs that were assigned')
    for i, diag in enumerate(ordered_diags):
        print(diag, labels[i])
        diag.label = labels[i]

    return ordered_diags


def get_threshold(panels):
    """ Get's a basic threshold value from area of all panels"""
    return 1.5 * np.mean([panel.area for panel in panels])


def get_labels_and_diagrams_k_means_clustering(panels):
    """ Splits into labels and diagrams using kmeans clustering of area"""

    all_params = np.array([[panel.perimeter / panel.area, panel.perimeter] for panel in panels])
    km = KMeans(n_clusters=2)
    clusters = km.fit(all_params)

    group_1, group_2 =[], []

    for i, cluster in enumerate(clusters.labels_):
        print(cluster)
        if cluster == 0:
            group_1.append(panels[i])
        else:
            group_2.append(panels[i])

    if np.mean([panel.area for panel in group_1]) > np.mean([panel.area for panel in group_2]):
        diags = group_1
        labels = group_2
    else:
        diags = group_2
        labels = group_1

    # Convert to appropriate types
    labels = [Label(label.left, label.right, label.top, label.bottom, label.tag) for label in labels if label.area > 8]
    diags = [Diagram(diag.left, diag.right, diag.top, diag.bottom, diag.tag) for diag in diags]
    return labels, diags


def get_labels_and_diagrams_from_threshold(panels):
    """ Separates into diagrams and lables based on a threshold
    CURRENTLY NOT IMPLEMENTED
    """

    # TODO : Change thresholding logic to a whitespace ratio from orig image
    area_mean = np.mean([panel.area for panel in panels])
    area_std = np.std([panel.area for panel in panels])

    # Identifying labels from values that are one std less than the mean
    labels = [panel for panel in panels if panel.area < (area_mean - 0.25*area_std)]
    diagrams = [panel for panel in panels if panel.area >= (area_mean - 0.25*area_std)]

    return labels, diagrams


def get_threshold_width_height(panels):
    """ Creates a thresholdbased on the average width and height of all images"""

    # NB : Not implemented anywhere currently

    width_avg = np.mean([panel.width for panel in panels])
    height_avg = np.mean([panel.width for panel in panels])

    width_std = np.std([panel.width for panel in panels])
    height_std = np.std([panel.width for panel in panels])


def classify_kruskal(panels):
    """ Classifies diagrams and labels for panels using Kruskals algorithm
        Weights are determined as the distance between panel centroids
    """

    diags = []
    labels = []
    used_tags = []

    sorted_edges = kruskal(panels)
    thresh = get_threshold(panels)

    for edge in sorted_edges:

        for panel in panels:
            if panel.tag == edge[0]:
                p1 = panel
            elif panel.tag == edge[1]:
                p2 = panel

        if p1.tag in used_tags or p2.tag in used_tags:
            pass
        elif p2.area < thresh < p1.area:
            diags.append(p1)
            labels.append(p2)
            used_tags.extend([p1.tag, p2.tag])

        elif p1.area < thresh < p2.area:
            diags.append(p2)
            labels.append(p1)
            used_tags.extend([p1.tag, p2.tag])
        elif p1.area < thresh and p2.area < thresh:
            print('Error - both panels are below the threshold')

    print('Assigned Pairs')
    for i, diag in enumerate(diags):
        print(diag, labels[i])

    return diags, labels


def label_kruskal(diags, labels):
    """ Test function for labelling by kruskals algorithm
    CURRENTLY NOT IN USE - labelling done through
    TODO : Implement in the main classify_kruskal if this works

    """

    diags = [Diagram(diag.left, diag.right, diag.top, diag.bottom, diag.tag) for diag in diags]

    for i, diag in enumerate(diags):
        diag.label = labels[i]

    return diags


def order_by_area(panels):
    """ Returns a list of panel objects ordered by area"""

    def get_area(panel):
        return panel.area

    panels.sort(key=get_area)
    return panels


def find_next_merge_candidate(thresh, ordered_panels):
    """
    Iterates through all candidate panels, merging when criteria is matched
    :param thresh: Used to determine that a panel constitutes a label
    :param ordered_panels: an ordered list of all panels in the image
    :return:
    """

    for a, b in list(set(itertools.combinations(ordered_panels, 2))):

        # Check panels lie in roughly the same line, that they are of label size and similar height
        if abs(a.center[1] - b.center[1]) < 1.5 * a.height \
                and abs(a.height - b.height) < a.height \
                and a.area < thresh and b.area < thresh:

            # Check that the distance between the edges of panels is not too large
            if (a.center[0] > b.center[0] and a.left - b.right > 0.5 * a.width) or (
                    a.center[0] < b.center[0] and b.left - a.right > 0.5 * a.width):
                merged_rect = merge_rect(a, b)
                if a in ordered_panels: ordered_panels.remove(a)
                if b in ordered_panels: ordered_panels.remove(b)
                ordered_panels.append(merged_rect)  # NB : merged rectangles will be the last to be compared

                ordered_panels = find_next_merge_candidate(thresh, ordered_panels)

    return ordered_panels


def is_unque_panel(a, b):
    """Checks whether a panel is unique"""

    if a.left == b.left and a.right == b.right \
            and a.top == b.top and a.bottom == b.bottom:
        return True
    else:
        return False


def merge_loop_horizontal(panels):
    """Goes through the loop for merging."""

    output_panels = []
    blacklisted_panels = []
    done = True

    for a, b in itertools.combinations(panels, 2):

        # Check panels lie in roughly the same line, that they are of label size and similar height
        if abs(a.center[1] - b.center[1]) < 1.5 * a.height \
                and abs(a.height - b.height) < a.height:

            # Check that the distance between the edges of panels is not too large
            if (0 < a.left - b.right < max(a.width, b.width))or (0 < (b.left - a.right) < max(a.width, b.width)):

                merged_rect = merge_rect(a, b)
                merged_panel = Panel(merged_rect.left, merged_rect.right, merged_rect.top, merged_rect.bottom, 0)
                output_panels.append(merged_panel)
                blacklisted_panels.extend([a, b])
                done = False

    print('Length of blacklisted : %s' % len(blacklisted_panels))
    print('Length of output panels : %s' % len(output_panels))

    for panel in panels:
        if panel not in blacklisted_panels:
            output_panels.append(panel)

    output_panels = relabel_panels(output_panels)
    return output_panels, done


def merge_loop_vertical(panels):
    """ Merding vertical panels within threshold"""
    output_panels = []
    blacklisted_panels = []

    # Merging labels that are in close proximity vertically
    for a, b in itertools.combinations(panels, 2):

        if abs(a.center[0] - b.center[0]) < 0.5 * max(a.width, b.width) and abs(a.center[1] - b.center[1]) < 3 * max(a.height, b.height) \
                and abs(a.height - b.height) < 0.3 * max(a.height, b.height) and abs(a.width - b.width) < 2 * max(a.width, b.width):

            merged_rect = merge_rect(a, b)
            merged_panel = Panel(merged_rect.left, merged_rect.right, merged_rect.top, merged_rect.bottom, 0)
            output_panels.append(merged_panel)
            blacklisted_panels.extend([a, b])

    for panel in panels:
        if panel not in blacklisted_panels:
            output_panels.append(panel)

    output_panels = relabel_panels(output_panels)

    print(output_panels)
    return output_panels


def merge_overlap(a, b):
    """ Checks whether panels a and b overlap. If they do, returns new merged panel"""

    if a.overlaps(b) or b.overlaps(a):
        return merge_rect(a, b)


def get_one_to_merge(all_combos, panels):
    """Returns the updated panel list once a panel needs to be merged"""

    for a, b in all_combos:

        overlap_panel = merge_overlap(a, b)
        if overlap_panel is not None:
            merged_panel = Panel(overlap_panel.left, overlap_panel.right, overlap_panel.top, overlap_panel.bottom, 0)
            panels.remove(a)
            panels.remove(b)
            panels.append(merged_panel)
            return panels, False

    return panels, True


def convert_panels_to_labels(panels):
    """ Converts a list of panels to a list of labels"""

    # TODO : Implement this whenever this conversion is made
    return [Label(panel.left, panel.right, panel.top, panel.bottom, panel.tag) for panel in panels]


def merge_all_overlaps(panels):
    """ Merges all overlapping rectangles together"""

    all_merged = False

    while all_merged is False:
        all_combos = list(itertools.combinations(panels, 2))
        panels, all_merged = get_one_to_merge(all_combos, panels)

    output_panels = relabel_panels(panels)
    return output_panels, all_merged


def merge_label_horizontally(merge_candidates):
    """ Try to merge horizontally by brute force method"""

    done = False

    # Identifies panels within horizontal merging criteria
    while done is False:
        ordered_panels = order_by_area(merge_candidates)
        merge_candidates, done = merge_loop_horizontal(ordered_panels)

    merge_candidates, done = merge_all_overlaps(merge_candidates)
    return merge_candidates


def merge_labels_vertically(merge_candidates):
    """ Try to merge vertically using brute force method"""

    # Identifies panels within horizontal merging criteria
    ordered_panels = order_by_area(merge_candidates)
    merge_candidates = merge_loop_vertical(ordered_panels)

    merge_candidates, done = merge_all_overlaps(merge_candidates)
    return merge_candidates


def merge_rect(rect1, rect2):
    """ Merges rectangle with another, such that the bounding box enclose both

    :param Rect rect1: A rectangle
    :param Rect rect2: Another rectangle
    :return: Merged rectangle
    """

    left = min(rect1.left, rect2.left)
    right = max(rect1.right, rect2.right)
    top = min(rect1.top, rect2.top)
    bottom = max(rect1.bottom, rect2.bottom)
    return Rect(left, right, top, bottom)

def get_duplicate_labelling(labelled_diags):
    """ Returns a set of diagrams which share a label"""

    failed_diag_label = set(diag for diag in labelled_diags if not diag.label)
    filtered_labelled_diags = [diag for diag in labelled_diags if diag not in failed_diag_label]

    # Identifying cases with the same label:
    for a, b in itertools.combinations(filtered_labelled_diags, 2):
        if a.label == b.label:
            failed_diag_label.add(a)
            failed_diag_label.add(b)

    return failed_diag_label


def label_diags(labels, diags, fig_bbox, rate=1):
    """ Pair all diags to labels using assign_label_to_diag"""

    # Sort diagrams from largest to smallest
    diags.sort(key=lambda x: x.area, reverse=True)
    initial_sorting = [assign_label_to_diag(diag, labels, fig_bbox) for diag in diags]

    # Identify failures by the presence of duplicate labels
    failed_diag_label = get_duplicate_labelling(initial_sorting)

    if len(failed_diag_label) == 0:
        return initial_sorting

    # Find average position of label relative to diagram (NSEW)
    successful_diag_label = [diag for diag in diags if diag not in failed_diag_label]

    if len(successful_diag_label) == 0:

        #Attempt looking 'South' for all diagrams (most common realtive label position)
        altered_sorting = [assign_label_to_diag_postprocessing(diag, labels, 'S', fig_bbox) for diag in failed_diag_label]
        if len(get_duplicate_labelling(altered_sorting)) != 0:
            return initial_sorting
        else:
            return altered_sorting

    # Compass positions of labels relative to diagram
    diag_compass = [diag.compass_position(diag.label) for diag in successful_diag_label if diag.label]
    mode_compass = max(diag_compass, key=diag_compass.count)

    # Then, expand outwards in this direction for all failures.
    altered_sorting = [assign_label_to_diag_postprocessing(diag, labels, mode_compass, fig_bbox) for diag in failed_diag_label]
    return altered_sorting + successful_diag_label


def assign_label_to_diag(diag, labels, fig_bbox, rate=1):
    """ Iteratively expands the bounding box of diagram until it reaches a label"""

    probe_rect = Rect(diag.left, diag.right, diag.top, diag.bottom)
    found = False
    max_threshold_width = fig_bbox.width
    max_threshold_height = fig_bbox.height

    while found is False and (probe_rect.width < max_threshold_width or probe_rect.height < max_threshold_height):
        # Increase border value each loop
        probe_rect.right = probe_rect.right + rate
        probe_rect.bottom = probe_rect.bottom + rate
        probe_rect.left = probe_rect.left - rate
        probe_rect.top = probe_rect.top - rate

        for label in labels:
            if probe_rect.overlaps(label):
                found = True
                print(diag.tag, label.tag)
                diag.label = label
    return diag


def assign_label_to_diag_postprocessing(diag, labels, direction, fig_bbox, rate=1):
    """ Iteratively expands the bounding box of diagram in the specified direction"""

    probe_rect = Rect(diag.left, diag.right, diag.top, diag.bottom)
    found = False

    def label_loop():

        for label in labels:
            # Only accepting labels in the average direction
            if diag.compass_position(label) != direction:
                pass
            elif probe_rect.overlaps(label):
                print(diag.tag, label.tag)
                diag.label = label
                return True

        return False

    # Increase border value each loop
    if direction == 'E':
        while found is False and probe_rect.right < fig_bbox.right:
            probe_rect.right = probe_rect.right + rate
            found = label_loop()

    elif direction == 'S':
        while found is False and probe_rect.bottom < fig_bbox.bottom:
            probe_rect.bottom = probe_rect.bottom + rate
            found = label_loop()

    elif direction == 'W':
        while found is False and probe_rect.left > fig_bbox.left:
            probe_rect.left = probe_rect.left - rate
            found = label_loop()

    elif direction == 'N':
        while found is False and probe_rect.top > fig_bbox.top:
            probe_rect.top = probe_rect.top - rate
            found = label_loop()
    else:
        return diag

    return diag


def read_all_labels(fig, diags):
    """ Reads the values of all labels"""

    for diag in diags:
        diag.label.text = read_label(fig, diag.label)
        print(diag.tag, diag.label.text)
    return diags


def read_all_diags(fig, diags):
    """ Resolves all diagrams to smiles"""

    for diag in diags:
        diag.smile = read_diagram(fig, diag)
        print(diag.tag, diag.smile)
    return diags

def read_diag_text(fig, diag, whitelist=LABEL_WHITELIST):
    """ Reads a diagram using OCR and returns the textual OCR objects"""
    img = convert_greyscale(fig.img)
    cropped_img = crop(img, diag.left, diag.right, diag.top, diag.bottom)
    text = get_text(cropped_img, x_offset=diag.left, y_offset=diag.top, psm=PSM.SINGLE_BLOCK, whitelist=whitelist)
    tokens = get_words(text)
    return tokens


def read_label(fig, label, whitelist=LABEL_WHITELIST):
    """ Reads a label paragraph objects using ocr

    :param numpy.ndarray img: Input unprocessedimage
    :param Label label: Label object containing appropriate bounding box

    :rtype List[List[str]]
    """

    img = convert_greyscale(fig.img)
    cropped_img = crop(img, label.left, label.right, label.top, label.bottom)
    text = get_text(cropped_img, x_offset=label.left, y_offset=label.top, psm=PSM.SINGLE_BLOCK, whitelist=whitelist)
    raw_sentences = get_sentences(text)

    if len(raw_sentences) is not 0:
        # Tag each sentence
        tagged_sentences = [Sentence(sentence, parsers=[LabelParser()]) for sentence in raw_sentences]
    else:
        tagged_sentences = []
    label.text = tagged_sentences
    return label


def read_diagram(fig, diag):
    """ Converts diagram to SMILES using OSRA"""

    cropped_img = crop(fig.img, diag.left, diag.right, diag.top, diag.bottom)

    # Write crop to temporary file
    temp_name = 'temp.png'
    imsave(temp_name, cropped_img)
    bash_command = "osra -r 300 -p ./" + temp_name

    # Use OSRA to decode SMILES
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    os.remove(temp_name)

    # Format result
    output = output.decode("utf-8")
    all_results = output.split('\n')  # Split into individual diagrams (should only be one anyway)
    results = all_results[0].split(' ')

    if results == ['']:
        return '', ''
    else:
        return results[0], results[1][:-2]


def read_diagram_pyosra(diag):
    """ Converts a diagram to SMILES using PYOSRA"""

    # Save a temp image
    temp_img_fname = 'osra_temp.jpg'
    imsave(temp_img_fname, diag.fig.img)

    # Run osra on temp image
    smile = osra_rgroup.read_diagram(temp_img_fname)

    imdel(temp_img_fname)

    smile = clean_output(smile)
    return smile


def clean_output(smile):
    """ Remove whitespace and newline characters"""

    smile = smile.replace(' ', '')
    return smile.replace('\n', '')


def get_diag_and_label(img):
    """ Segments images into diagram-label pairs
    CURRENTLY NOT USED
    :param numpy.ndarray img: Input image
    :return: Binary image.
    :rtype: numpy.ndarray
    """
    num_lbls = 1000 # Arbitrarily large number
    int_img = None
    while num_lbls > 6: # TO DO : automatically determine 6
        img = morphology.binary_dilation(img, selem=morphology.disk(2))
        int_img = img.astype(int)
        labels, num_lbls = morphology.label(int_img, return_num=True)
        print(num_lbls)
    return int_img


def get_img_boundaries(img, left=None, right=None, top=None, bottom=None):
    """ Gets the boundaries of a numpy image
    
    :param numpy.ndarray img: Input image.
    :param int left: Left crop.
    :param int right: Right crop.
    :param int top: Top crop.
    :param int bottom: Bottom crop.
    :return: list(int left, int right, int top, int bottom)""" 

    # TODO : Use this inside crop
    height, width = img.shape[:2]

    left = max(0, 0 if left is None else left )
    right = min(width, width if right is None else right)
    top = max(0, 0 if top is None else top)
    bottom = min(height, height if bottom is None else bottom)
    return left, right, top, bottom
