# -*- coding: utf-8 -*-
"""
Image processing validation metrics

========

A toolkit of validation metrics for determining reliability of output

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

from .actions import crop
import numpy as np
import re
#import pybel


def is_false_positive(label_smile_tuple):
    """ Identifies failures from absernce of labels and incomplete / invalid smiles

    :rtype bool
    :returns : True if result is a false positive
    """

    label_candidates, smile = label_smile_tuple[0], label_smile_tuple[1]
    # Remove results without a label
    if len(label_candidates) == 0:
        return True

    # Remove results containing the wildcard character in the SMILE
    if '*' in smile:
        return True

    return False


def pixel_ratio(fig, diag):
    """ Calculates the ratio of 'on' pixels to bbox area for binary image

    :param numpy.ndarray img: Input image
    """

    cropped_img = crop(fig.img, diag.left, diag.right, diag.top, diag.bottom, padding=10)
    ones = np.count_nonzero(cropped_img)
    all_pixels = np.size(cropped_img)
    return ones / all_pixels


def total_pixel_ratio(fig, diags):
    """ Calculate the average ratio of on to off for entire diagram """

    ratios = [pixel_ratio(fig, diag) for diag in diags]
    avg = sum(ratios) / len(ratios)
    return avg


def diagram_to_image_area_ratio(fig, diags):
    """ Calculate the ratio of diagram number to total image area"""

    img = fig.img
    diag_no = len(diags)
    img_size = img.size

    return diag_no / img_size

def avg_diagram_area_to_image_area(fig, diags):
    """ Calculate ratio of average diagram area to total image area"""

    cropped_diags_img = [crop(fig.img, diag.left, diag.right, diag.top, diag.bottom, padding=10) for diag in diags]
    diags_size = [img.size for img in cropped_diags_img]
    avg_diag_size = sum(diags_size) / len(diags_size)
    return avg_diag_size / fig.img.size

def pybel_format(text):
    """
    Checks that a smiles is chemically valid
    """

    if '*' not in text: # Removes wildcards
        try:
            smile = pybel.readstring('smi', text).write("can")
            return re.sub('[\t\n]+', '', smile)
        except:
            print('Invalid smile %s' % text)
    return None

def format_all_smiles(diags):
    """ Checks list of smiles are valid"""
    formatted_diags = []
    for diag in diags:
        formatted = pybel_format(diag.smile)
        if formatted is not None:
            diag.smile = formatted
            formatted_diags.append(diag)

    return formatted_diags

"""
############################################################
The following code is from Ganesh, needs citation if used.
############################################################
"""


def CleanSMI(self):


     """
         My jobs is to  remove all the non standard characters that might slip in to the smiles string.
         :return type: str
     """

     return re.sub(r'[<>%\\/?\|]+', '', self.smile)


def HardValidSMI(self):

    """
    A rule based function to validate a given smile string.
    Return type: Boolean
    True: If a conjugated polymer is found.
    False: Charges, Ions and No Conjugated regions found.
    """

    mysmile = self.CleanSMI()
    illegalstring = re.search(r'\\|/|\*|Fe|\+\+|\.|\|', mysmile)  # --> Sanity check!

    if illegalstring:

        return False
    else:
        cansmile = pybel.readstring("smi", mysmile).write("can")

        # Matches up to 3 word characters, followed by a +,-,. or a number, all enlosed n square brackets
        # Also mactches 2 word characters in a row
        # Also matches 2 full stops in a row ..
        # Also matches the wildcards surrounded by brackets
        regex_string = r'\[\w{1,3}[\+-\.\d]+\]|\[\w{2}\]|\.|\(\*\)'

        match = re.search(regex_string, cansmile)

    if match:

        return False
    else:

        # Matches a-z or capital word character
        # Followed by one or more numbers
        # Followed by one of a-z or D-Z or capital word characters, as many times as needed, followed by a digit
        conj_match_string = '[a-z\W]\d+[a-zD-Z\W]+\d'

        conjuated = re.search(conj_match_string, mysmile)  # r'[a-z\W]\d+[\w\W]+\d' or use (r'[a-z\W]\d+[a-zD-Z\W]+\d',mysmile) )

    if conjuated:

        return True

    else:

        return False

@staticmethod
def BlackList(localsmi):
    """
    A function to read a blacklist of smiles that will crash rdkit
    """
    with open('black_list.csv', 'r') as csvfile:
        content =  csvfile.readlines()
        content = content[0].replace("'","").split(',')
        content = [data.strip('\n') for data in content]
        if localsmi in content:
            return True
        else:
            return False




