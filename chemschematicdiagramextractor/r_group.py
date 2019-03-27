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

import osra_rgroup
import cirpy
import itertools

from . import io
from . import actions
from .ocr import ASSIGNMENT, SEPERATORS, CONCENTRATION

log = logging.getLogger(__name__)

BLACKLIST_CHARS = ASSIGNMENT + SEPERATORS + CONCENTRATION


def detect_r_group(diag):
    """ Determines whether a label represents an R-Group structure, and if so gives the variable and value

    # TODO : docstring
    """

    sentences = diag.label.text
    for sentence in sentences:
        var_value_pairs = []
        for i, token in enumerate(sentence.tokens):
            if token.text is '=':
                print('Found R-Group descriptor %s' % token.text)
                if i > 0:
                    print('Variable candidate is %s' % sentence.tokens[i-1] )
                if i < len(sentence.tokens):
                    print('Value candidate is %s' % sentence.tokens[i+1])

                if 0 < i < len(sentence.tokens):
                    variable = sentence.tokens[i - 1]
                    value = sentence.tokens[i + 1]
                    var_value_pairs.append((variable, value))

            elif token.text == 'or' and var_value_pairs:
                print('"or" keyword detected. Assigning value to previous R-group variable...')

                # Identify the most recent var_value pair
                variable = var_value_pairs[-1][0]
                value = sentence.tokens[i + 1]
                var_value_pairs.append((variable, value))

        var_value_label_triplets = get_label_candiates(sentence, var_value_pairs)

        var_value_label_triplets = standardize_values(var_value_label_triplets)

        # TODO : Add logic here to detect all other alphanumerics using re package
        diag.label.add_r_group_variables(var_value_label_triplets)

    return diag


def get_label_candiates(sentence, var_value_pairs, blacklist_chars=BLACKLIST_CHARS, blacklist_words=['or']):
    ''' Extracts label candidates from a sentence that ontains r-groups'''

    candidates = [token for token in sentence.tokens if token.text not in blacklist_chars]
    vars_and_values = []
    for var, value in var_value_pairs:
        vars_and_values.append(var)
        vars_and_values.append(value)

    print(candidates)

    output = []

    for var, value in var_value_pairs:
        label_cands = []
        for token in candidates:
            if token not in vars_and_values:
                label_cands.append(token)

        output.append((var, value, label_cands))

    return output


def get_rgroup_smiles(diag, fig, cleanchars='()'):
    """ Uses modified version of OSRA to get SMILES for multiple """

    # Save a temp image
    io.imsave('r_group_temp.jpg', actions.crop(fig.img, diag.left, diag.right, diag.top, diag.bottom))

    osra_input = []
    label_cands = []
    smiles = []

    #Format the extracted rgroup
    for tokens in diag.label.r_group:
        token_dict = {}
        for token in tokens:


            #value = clean_chars(token[1].text, cleanchars)
            token_dict[token[0].text] = token[1].text
            #smiles.append(resolve_structure(token[1].text))

        # Assigning var-var cases to true value if found (eg. R1=R2=H)
        for a, b in itertools.combinations(token_dict.keys(), 2):
            if token_dict[a] == b:
                token_dict[a] = token_dict[b]


        osra_input.append(token_dict)
        label_cands.append(tokens[0][2])
        print(" Smiles resolved were:  %s" % smiles)

    # TODO : Attempt to resolve compound values using cirpy / OPSIN

    # Run osra on temp image
    smiles = osra_rgroup.hack_osra_process_image(osra_input, input_file="r_group_temp.jpg")

    io.imdel('r_group_temp.jpg')

    smiles = [smile.replace('\n', '') for smile in smiles]

    labels_and_smiles = []
    for i, smile in enumerate(smiles):
        labels_and_smiles.append((label_cands[i], smile))

    return labels_and_smiles


def clean_chars(value, cleanchars):
    """ Remove chars for cleaning"""

    for char in cleanchars:
        value = value.replace(char, '')

    return value


def resolve_structure(compound):
    """ Resolves compound structure using cirpy"""

    smiles = cirpy.resolve(compound, 'smiles')

    return smiles


def standardize_values(var_value_label_triplets):
    """ Converts values to a format compatible with pyosra"""

    # List of tuples pairing multiple definitions to the appropriate SMILES string
    alkyls = [('C', ['methyl']),
              ('CC', ['ethyl'])]

    for triplet in var_value_label_triplets:
        value = triplet[1]
        for alkyl in alkyls:
            if value.lower() in alkyl[1]:
                triplet[1] = alkyl[0]

    return var_value_label_triplets


