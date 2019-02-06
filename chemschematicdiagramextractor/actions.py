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
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage import morphology
from skimage.feature import canny
import skimage.io as skio
from skimage.filters import sobel
from skimage.util import pad, crop
from skimage.util import crop as crop_skimage
from skimage.morphology import watershed, closing, binary_closing, disk, square, rectangle
from skimage.measure import regionprops
import tempfile
import subprocess
import os
import itertools
import copy

from scipy import ndimage as ndi
from matplotlib import pyplot as plt

from .model import Panel, Diagram, Label, Rect, Graph
from .ocr import get_text, get_lines, PSM, LABEL_WHITELIST
from .io import img_as_pil , imsave # for debugging

log = logging.getLogger(__name__)

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


def binary_close(fig, size=16, ratio=2):
    """ Joins unconnected pixel by dilation and erosion"""
    #selem = disk(size)
    width = size*ratio
    selem = rectangle(width, size)

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

    return grey_img

def get_bounding_box(fig):
    """ Gets the bounding box of each segment"""
    panels =[]
    regions = regionprops(fig.img)
    for region in regions:
        y1,x1,y2,x2 = region.bbox
        panels.append(Panel(x1,x2,y1,y2, region.label - 1)) # Sets tags to start from 0

    return panels

def get_repeating_unit(panels, fig):
    """ Identifies 'n' labels as repeating unit identifiers"""

    ns = []

    thresh = get_threshold(panels)
    diags = (panel for panel in panels if panel.area > thresh)

    for diag in diags:
        for panel in panels:
            if diag.overlaps(panel) and (panel.center_px[0] - diag.center_px[0]) > 0:
                repeating_units = [word for sentence in read_label(fig, panel) for word in sentence if 'n' in word]
                if repeating_units:
                    ns.append(panel)
                    diag.repeating = True

    panels = [panel for panel in panels if panel not in ns]
    panels = relabel_panels(panels)

    return panels


def relabel_panels(panels):
    for i, panel in enumerate(panels):
        panel.tag = i

    return panels

def segment(fig):
    """ Segments image """

    bin_fig = binarize(fig)

    closed_fig = binary_close(bin_fig)

    out_fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(closed_fig.img)
    plt.show()

    fill_img = binary_floodfill(closed_fig)
    tag_img, no_tagged = binary_tag(fill_img)
    panels = get_bounding_box(tag_img)

    return panels

def preprocessing(panels, fig):

    # Pre-processing filtering
    panels = get_repeating_unit(panels, fig)
    panels = merge_labels_horizontally(panels)
    panels = merge_labels_vertically(panels)
    return panels


def classify(panels):
    """ Classifies diagrams and labels for panels  """

    diags = []
    labels = []

    areas = [panel.area for panel in panels]
    mean = np.mean(areas) # Threshold for classification
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

def get_threshold(panels):
    """ Gets threshold for diagrams vs labels from panel areas"""

    # TODO : Change thresholding logic to a whitespace ratio from orig image
    areas = [panel.area for panel in panels]
    return 0.6*np.mean(areas) # Threshold for classification

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

            print('broken')

    print('Pairs that were assigned')
    for i, diag in enumerate(diags):
        print (diag, labels[i])

    return diags, labels


def label_kruskal(diags, labels):
    """ Test function for labelling by kruskals algorithm
    TODO : Implement in the main classify_kruskal if this works

    """

    diags = [Diagram(diag.left, diag.right, diag.top, diag.bottom, diag.tag) for diag in diags]

    for i, diag in enumerate(diags):
        diag.label = labels[i]

    return diags


def merge_labels_horizontally(panels):
    """ Identifies labels to merge from vertical proximity and length"""

    output_panels = []
    blacklisted_panels = []

    thresh = get_threshold(panels) # Could use a different threshold function for horizontal?

    # Merging labels that are in close proximity vertically
    for a, b in itertools.combinations(panels, 2):
        # Check that center of label is approximately 1.5*width of this box from the center of this box
        # Check that height of labels are about the same
        # Check that width of the boxes are comparable
        # Check whitespace ratio? We do have the original figure (unbinarized etc) here now...

        # if abs(a.center[0] - b.center[0]) < 1.5*a.width and abs(a.center[1] - b.center[1]) < 2*a.height \
        #         and abs(a.height - b.height) < 0.2*a.height and abs(a.width - b.width) < 2*a.width \
        #         and a.area < thresh and b.area < thresh:

        if abs(a.center[1] - b.center[1]) < 0.5 * a.height: # Check labels are of similar height

            if a.center[0] > b.center[0]: # Check which box is of greater x
                if (a.left - b.right < a.width) or (a.left - b.right < b.width):

                    merged_rect = merge_rect(a, b)
                    merged_panel = Panel(merged_rect.left, merged_rect.right, merged_rect.top, merged_rect.bottom, 0)
                    output_panels.append(merged_panel)
                    blacklisted_panels.extend([a, b])

            else:
                if (b.left - a.right < a.width) or (b.left - a.right < b.width):

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



def merge_labels_vertically(panels):
    """ Identifies labels to merge from vertical proximity and length"""

    output_panels = []
    blacklisted_panels = []

    thresh = get_threshold(panels)

    # Merging labels that are in close proximity vertically
    for a, b in itertools.combinations(panels, 2):
        # Check that center of label is approximately 1.5*height of this box from the center of this box
        # Check that height of labels are about the same
        # Check that width of the boxes are comparable
        # Check whitespace ratio? We do have the original figure (unbinarized etc) here now...

        # if abs(a.center[0] - b.center[0]) < 1.5*a.width and abs(a.center[1] - b.center[1]) < 2*a.height \
        #         and abs(a.height - b.height) < 0.2*a.height and abs(a.width - b.width) < 2*a.width \
        #         and a.area < thresh and b.area < thresh:

        if abs(a.center[0] - b.center[0]) < 1.5 * a.width and abs(a.center[1] - b.center[1]) < 3 * a.height \
                and abs(a.height - b.height) < 0.3 * a.height and abs(a.width - b.width) < 2 * a.width \
                and a.area < thresh and b.area < thresh:

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



def merge_labels_kruskal(panels):
    """ Identifies labels to merge from sorted edges"""

    output_panels = []
    blacklisted_panels = []

    edges = kruskal(panels)
    thresh = get_threshold(panels)

    for edge in edges:
        # Could simplify this loop using dict?
        for panel in panels:
            if panel.tag == edge[0]:
                p1 = panel
            elif panel.tag == edge[1]:
                p2 = panel

        if p1.area < thresh and p2.area < thresh:
            merged_rect = merge_rect(p1, p2)
            merged_panel = Panel(merged_rect.left, merged_rect.right, merged_rect.top, merged_rect.bottom, 0)
            output_panels.append(merged_panel)
            blacklisted_panels.extend([p1, p2])

    for panel in panels:
        if panel not in blacklisted_panels:
            output_panels.append(panel)

    output_panels = relabel_panels(output_panels)

    return output_panels

def merge_rect(rect1, rect2):
    """ Merges rectangle with another, such that the bounding box enclose both

    :param Rect other_rect: Another rectangle
    :return: Merged rectangle"""

    left = min(rect1.left, rect2.left)
    right = max(rect1.right, rect2.right)
    top = min(rect1.top, rect2.top)
    bottom = max(rect1.bottom, rect2.bottom)

    return Rect(left, right, top, bottom)


def label_diags(diags, labels, rate=1):
    """ Pair all diags to labels using assign_label_to_diag"""

    return [assign_label_to_diag(diag, labels, rate) for diag in diags]


def assign_label_to_diag(diag, labels, rate=1):
    """ Iteratively expands the bounding box of diagram until it reaches a label"""

    probe_rect = Rect(diag.left, diag.right, diag.top, diag.bottom)
    found=False
    # TODO : Add thresholds for right, bottom, left and top

    while found==False :
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


def read_label(fig, label, whitelist=LABEL_WHITELIST):
    """ Reads a label paragraph objects using ocr

    :param numpy.ndarray img: Input unprocessedimage
    :param Label label: Label object containing appropriate bounding box

    :rtype List[List[str]]
    """

    # TODO : Add post-processing check to see if the majority of returned characters wer letters / numbers. If above threshold, redo with changed 'whitelist' characters
    img = convert_greyscale(fig.img)
    cropped_img = crop(img, label.left, label.right, label.top, label.bottom)
    text = get_text(cropped_img, x_offset=label.left, y_offset=label.top, psm=PSM.SINGLE_BLOCK, whitelist=whitelist)
    label_text = get_lines(text)

    sentences = []

    if len(label_text) is not 0:
        for line in label_text:
            words = []
            for word in line:
                words.append(word.text)
            sentences.append(words)
    return sentences

def read_diagram(fig, diag):
    """ Converts diagram to SMILES using OSRA"""

    cropped_img = crop(fig.img, diag.left, diag.right, diag.top, diag.bottom)
    res = cropped_img.size
    # skio.imshow(cropped_img)
    # plt.show()

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

def get_diag_and_label(img):
    """ Segments images into diagram-label pairs

    :param numpy.ndarray img: Input image
    :return: Binary image.
    :rtype: numpy.ndarray
    """
    num_lbls = 1000
    while num_lbls > 6: # TO DO : automatically determine 6
        img = morphology.binary_dilation(img, selem=morphology.disk(2))# TODO: Dynamic selem size?
        int_img = img.astype(int)
        labels, num_lbls = morphology.label(int_img, return_num=True)
        print(num_lbls)
    return int_img

def crop(img, left=None, right=None, top=None, bottom=None, padding=0):
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

    #x_padding = int(np.around(padding))
    #y_padding = int(np.around(padding))

    left = max(0, 0 if left is None else left )
    right = min(width, width if right is None else right)
    top = max(0, 0 if top is None else top)
    bottom = min(height, height if bottom is None else bottom)
    img =  img[top: bottom, left : right ]
    #pad_img = np.pad(img, padding, mode='constant')
    return img

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
