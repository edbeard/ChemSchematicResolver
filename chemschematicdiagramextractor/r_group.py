# -*- coding: utf-8 -*-
"""
Scripts for identifying R-Group structure diagrams

========
author: Ed Beard
email: ejb207@cam.ac.uk

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

log = logging.getLogger(__name__)

import osra_rgroup
from . import io
from . import actions


def detect_r_group(diag):
    """ Determines whether a label represents an R-Group structure, and if so gives the variable and value

    # TODO : docstring
    """

    sentences = diag.label.text
    for sentence in sentences:
        var_variable_pairs = []
        for i, token in enumerate(sentence.tokens):
            if token.text is '=':
                print('Found R-Group descriptor %s' % token.text)
                if i > 0 :
                    print('Variable candidate is %s' % sentence.tokens[i-1] )
                if i < len(sentence.tokens):
                    print('Value candidate is %s' % sentence.tokens[i+1])

                if 0 < i < len(sentence.tokens):
                    variable = sentence.tokens[i - 1]
                    value = sentence.tokens[i + 1]
                    var_variable_pairs.append((variable, value))

        diag.label.add_r_group_variables(var_variable_pairs)


    return diag


def get_rgroup_smiles(diag, fig):
    """ Uses modified version of OSRA to get SMILES for multiple """

    # Save a temp image
    io.imsave('r_group_temp.jpg', actions.crop(fig.img, diag.left, diag.right, diag.top, diag.bottom))

    #Format the extracted rgroup
    osra_input = [{token[0].text: token[1].text} for tokens in diag.label.r_group for token in tokens]

    # Run osra on temp image
    smiles = osra_rgroup.hack_osra_process_image(osra_input, input_file="r_group_temp.jpg")

    io.imdel('r_group_temp.jpg')

    smiles = [smile.replace('\n', '') for smile in smiles]

    return smiles


# def tokenize_commas(sentences):
#     """ Takes a list of senteces, and reformats to include commas in tokenization"""
#
#     for sentence in sentences:
#         for i, token in enumerate(sentence):
#             if ',' in token.text:
