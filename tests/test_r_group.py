# -*- coding: utf-8 -*-
"""
test_parse
========

Test R-group resolution operations.

"""

from chemschematicresolver import r_group
from chemschematicresolver.model import RGroup

from chemdataextractor.doc.text import Sentence, Token, ChemSentenceTokenizer, ChemWordTokenizer, ChemLexicon, ChemAbbreviationDetector, ChemCrfPosTagger, CemTagger

import unittest


def do_resolve(comp):
    raw_smile = r_group.resolve_structure(comp)
    return raw_smile


class TestRgroup(unittest.TestCase):
    """ Test functios from the r_group.py module"""

    def test_resolve_structure_1(self):

        comp = '4-nitrophenyl'
        gold = '[O-][N+](=O)c1ccccc1'
        result = do_resolve(comp)
        self.assertEqual(gold, result)

    def test_resolve_structure_2(self):

        comp = '2-chloro-4-nitrophenol'
        gold = 'Oc1ccc(cc1Cl)[N+]([O-])=O'
        result = do_resolve(comp)
        self.assertEqual(gold, result)

    def test_resolve_structure_4(self):

        comp = 'Hexyl'
        gold = '[O-][N+](=O)c1cc(c(Nc2c(cc(cc2[N+]([O-])=O)[N+]([O-])=O)[N+]([O-])=O)c(c1)[N+]([O-])=O)[N+]([O-])=O'
        result = do_resolve(comp)
        self.assertEqual(gold, result)

    def test_duplicate_r_group_vars_in_one_sentence(self):

        sent = Sentence('A R1=H R2=NH B R1=H R2=C')

        # sent = Sentence(text=[Token('A', 0, 1), Token('R1', 2, 3), Token('=', 4, 5), Token('H', 6, 7),
        #                 Token('R2', 8, 9), Token('=', 10, 11), Token('NH', 12, 13),
        #                 Token('B', 14, 15), Token('R1', 16, 17), Token('=', 18, 19), Token('H', 20, 21),
        #                 Token('R2', 21, 22), Token('=', 23, 24), Token('H', 25, 26)],
        #                 start=0,
        #                 end=26,
        #                 sentence_tokenizer=ChemSentenceTokenizer(),
        #                 word_tokenizer=ChemWordTokenizer(),
        #                 lexicon=ChemLexicon(),
        #                 abbreviation_detector=ChemAbbreviationDetector(),
        #                 pos_tagger=ChemCrfPosTagger(),  # ChemPerceptronTagger()
        #                 ner_tagger=CemTagger()
        # )

        var_value_pairs = r_group.detect_r_group_from_sentence(sent)
        r_groups = r_group.get_label_candidates(sent, var_value_pairs)
        r_groups = r_group.standardize_values(r_groups)

        # Resolving positional labels where possible for 'or' cases
        r_groups = r_group.filter_repeated_labels(r_groups)

        # Separate duplicate variables into separate lists
        r_groups_list = r_group.separate_duplicate_r_groups(r_groups)

        output = []
        for r_groups in r_groups_list:
            output.append(r_group.convert_r_groups_to_tuples(r_groups))

    def test_r_group_simple_table(self):

        # Define a simple table structure
        table = [Sentence('R'), Sentence('1a CH3'), Sentence('1b Me')]

        output = r_group.resolve_r_group_grid(table)
        var, value, labels = output[0].convert_to_tuple()
        var2, value2, labels2 = output[1].convert_to_tuple()
        # tuple_output = [ (var.text, value.text, labels.text) for var, value, labels in output[0].convert_to_tuple()]
        self.assertEqual(var.text, 'R')
        self.assertEqual(value.text, 'CH3')
        self.assertEqual(labels[0].text, '1a')
        self.assertEqual(var2.text, 'R')
        self.assertEqual(value2.text, 'Me')
        self.assertEqual(labels2[0].text, '1b')

    def test_r_group_table(self):

        # Define a simple table structure
        table = [Sentence('R1 R2'), Sentence('1a CH3 C'), Sentence('1b Me Br')]

        output = r_group.resolve_r_group_grid(table)
        r_groups_list = r_group.separate_duplicate_r_groups(output)

        # Test the first r_group pair
        var1, value1, labels1 = r_groups_list[0][0].convert_to_tuple()
        var2, value2, labels2 = r_groups_list[0][1].convert_to_tuple()

        self.assertEqual(var1.text, 'R1')
        self.assertEqual(value1.text, 'CH3')
        self.assertEqual(labels1[0].text, '1a')
        self.assertEqual(var2.text, 'R2')
        self.assertEqual(value2.text, 'C')
        self.assertEqual(labels2[0].text, '1a')

        # Test the second r_group pair
        var1, value1, labels1 = r_groups_list[1][0].convert_to_tuple()
        var2, value2, labels2 = r_groups_list[1][1].convert_to_tuple()

        self.assertEqual(var1.text, 'R1')
        self.assertEqual(value1.text, 'Me')
        self.assertEqual(labels1[0].text, '1b')
        self.assertEqual(var2.text, 'R2')
        self.assertEqual(value2.text, 'Br')
        self.assertEqual(labels2[0].text, '1b')

    def test_r_group_assignment(self):
        """
        Test assignment of multiple lines
        """

        sentences = [Sentence('R1 = R2 = H'), Sentence('R1 = R2 = Ac')]
        out = []
        for sentence in sentences:
            r_groups = r_group.detect_r_group_from_sentence(sentence, indicator='=')
            r_groups = r_group.standardize_values(r_groups)

            # Resolving positional labels where possible for 'or' cases
            r_groups = r_group.filter_repeated_labels(r_groups)

            # Separate duplicate variables into separate lists
            r_groups_list = r_group.separate_duplicate_r_groups(r_groups)

            out.append(r_groups_list)

        self.assertEqual(out[0][0][0].var.text, 'R1')
        self.assertEqual(out[0][0][0].value.text, 'R2')

        self.assertEqual(out[0][0][1].var.text, 'R2')
        self.assertEqual(out[0][0][1].value.text, '[H]')

        self.assertEqual(out[1][0][0].var.text, 'R1')
        self.assertEqual(out[1][0][0].value.text, 'R2')

        self.assertEqual(out[1][0][1].var.text, 'R2')
        self.assertEqual(out[1][0][1].value.text, 'Ac')
