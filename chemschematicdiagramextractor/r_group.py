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
from .model import RGroup
from .ocr import ASSIGNMENT, SEPERATORS, CONCENTRATION

import re
from skimage.util import pad

from chemdataextractor.doc.text import Token

log = logging.getLogger(__name__)

BLACKLIST_CHARS = ASSIGNMENT + SEPERATORS + CONCENTRATION

# Regular Expressions
NUMERIC_REGEX = re.compile('^\d{1,3}$')
ALPHANUMERIC_REGEX = re.compile('^((d-)?(\d{1,2}[A-Za-z]{1,2}[′″‴‶‷⁗]?)(-d))|(\d{1,3})?$')


def detect_r_group(diag):
    """ Determines whether a label represents an R-Group structure, and if so gives the variable and value

    # TODO : docstring
    """

    sentences = diag.label.text
    for sentence in sentences:
        var_value_pairs = []  # Used to find variable - value pairs for extraction

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
                    var_value_pairs.append(RGroup(variable, value, []))

            elif token.text == 'or' and var_value_pairs:
                print('"or" keyword detected. Assigning value to previous R-group variable...')

                # Identify the most recent var_value pair
                variable = var_value_pairs[-1].var
                value = sentence.tokens[i + 1]
                var_value_pairs.append(RGroup(variable, value, []))

        # Process R-group values from '='
        r_groups = get_label_candiates(sentence, var_value_pairs)
        r_groups = standardize_values(r_groups)
          # TODO : Add logic here to detect all other alphanumerics using re package

        # Resolving positional labels where possible for 'or' cases
        r_groups = filter_repeated_labels(r_groups)

        # Separate duplicate variables into separate lists
        r_groups_list = separate_duplicate_r_groups(r_groups)

        for r_groups in r_groups_list:

            diag.label.add_r_group_variables(convert_r_groups_to_tuples(r_groups))

        # Process R-group values from 'or' if present

    return diag


def get_label_candiates(sentence, r_groups, blacklist_chars=BLACKLIST_CHARS, blacklist_words=['or']):
    """Extracts label candidates from a sentence that contains r-groups"""

    # TODO : Combine this logic into one line?
    # Remove irrelevant characters and blacklisted words
    candidates = [token for token in sentence.tokens if token.text not in blacklist_chars]
    candidates = [token for token in candidates if token.text not in blacklist_words]

    r_group_vars_and_values = []
    for r_group in r_groups:
        r_group_vars_and_values.append(r_group.var)
        r_group_vars_and_values.append(r_group.value)

    candidates = [token for token in candidates if token not in r_group_vars_and_values]

    for r_group in r_groups:
        var = r_group.var
        value = r_group.value
        label_cands = [candidate for candidate in candidates if candidate not in [var, value]]
        r_group.label_candidates = label_cands

    return r_groups


def filter_repeated_labels(r_groups):
    """ Checks if the same variable is present twice. If yes, this is an 'or' case so relative label assignment ensues"""

    # Identify 'or' variables
    or_vars = []
    vars = [r_group.var for r_group in r_groups]
    unique_vars = set(vars)
    for test_var in unique_vars:
        if vars.count(test_var) > 1:
            print('Identified "or" variable')
            or_vars.append(test_var)

    # Get label candidates for r_groups containing this:
    filtered_r_groups = [r_group for r_group in r_groups if r_group.var in or_vars]

    # If no duplicate r_group variables, exit function
    if len(filtered_r_groups) == 0:
        return r_groups

    remaining_r_groups = [r_group for r_group in r_groups if r_group.var not in or_vars]
    label_cands = filtered_r_groups[0].label_candidates #  Get the label candidates for these vars (should be the same)

    # Prioritizing alphanumerics for relative label assignment
    alphanumeric_labels = [label for label in label_cands if ALPHANUMERIC_REGEX.match(label.text)]

    output_r_groups = []

    # First check if the normal number of labels is the same
    if len(filtered_r_groups) == len(label_cands):
        for i in range(len(filtered_r_groups)):
            altered_r_group = filtered_r_groups[i]
            altered_r_group.label_candidates = [label_cands[i]]
            output_r_groups.append(altered_r_group)
        output_r_groups = output_r_groups + remaining_r_groups

    # Otherwise, check if alphanumerics match
    elif len(filtered_r_groups) == len(alphanumeric_labels):
        for i in range(len(filtered_r_groups)):
            altered_r_group = filtered_r_groups[i]
            altered_r_group.label_candidates = [alphanumeric_labels[i]]
            output_r_groups.append(altered_r_group)
        output_r_groups = output_r_groups + remaining_r_groups

    # Otherwise return with all labels
    else:
        output_r_groups = r_groups

    return output_r_groups



    # vars_and_values = []
    # for r_group in var_value_pairs:
    #     vars_and_values.append(r_group.var)
    #     vars_and_values.append(r_group.value)
    #
    # print(candidates)
    #
    # output = []
    #
    # for r_group in var_value_pairs:
    #     label_cands = []
    #     for token in candidates:
    #         if token not in vars_and_values:
    #             label_cands.append(token)
    #
    #     output.append((r_group.var, r_group.value, label_cands))
    #
    # return output

def get_rgroup_smiles(diag, cleanchars='()'):
    """ Uses modified version of OSRA to get SMILES for multiple """

    # Add some padding to image to help resolve characters on the edge
    padded_img = pad(diag.fig.img, ((5,5), (5,5), (0, 0)), mode='constant', constant_values=1)

    # Save a temp image
    io.imsave('r_group_temp.jpg', padded_img)

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

    smiles = [actions.clean_output(smile) for smile in smiles]

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


def convert_r_groups_to_tuples(r_groups):
    """ Converts a list of Rgroup model objets to Rgroup tuples"""

    return [r_group.convert_to_tuple() for r_group in r_groups]


def standardize_values(r_groups):
    """ Converts values to a format compatible with pyosra"""

    # List of tuples pairing multiple definitions to the appropriate SMILES string
    # TODO : SHould define these globally?
    alkyls = [('CH', ['methyl']),
              ('C2H', ['ethyl']),
              ('C3H', ['propyl']),
              ('C4H', ['butyl']),
              ('C5H', ['pentyl']),
              ('C6H', ['hexyl']),
              ('C7H', ['heptyl']),
              ('C8H', ['octyl']),
              ('C9H', ['nonyl']),
              ('C10H', ['decyl'])]

    for r_group in r_groups:
        value = r_group.value.text
        for alkyl in alkyls:
            if value.lower() in alkyl[1]:
                r_group.value = Token(alkyl[0], r_group.value.start, r_group.value.end, r_group.value.lexicon)

    return r_groups


def separate_duplicate_r_groups(r_groups):
    """ Separate duplicate R-group variables into separate lists"""

    if len(r_groups) is 0:
        return r_groups

    vars = [r_group.var for r_group in r_groups]
    unique_vars = list(set(vars))

    var_quantity_tuples = []
    vars_dict = {}
    output = []

    for var in unique_vars:
        var_quantity_tuples.append((var, vars.count(var)))
        vars_dict[var.text] = []

    equal_length = all(elem[1] == var_quantity_tuples[0][1] for elem in var_quantity_tuples)

    # If irregular, default behaviour is to just use one of the values
    if not equal_length:
        return r_groups

    # Populate dictionary for each unque variable
    for var in unique_vars:
        for r_group in r_groups:
            if var == r_group.var:
                vars_dict[var.text].append(r_group)

    print(vars_dict)

    for i in range(len(vars_dict[var.text])):
        temp = []
        for var in unique_vars:
            temp.append(vars_dict[var.text][i])
        output.append(temp)

    return output



