# -*- coding: utf-8 -*-
"""
test_validation
========

Test image processing on images from examples

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

log = logging.getLogger(__name__)

import chemschematicresolver.validate as val
import unittest


class TestValidation(unittest.TestCase):

    def test_remove_false_positives(self):

        self.assertTrue(val.is_false_positive(([], 'C1CCCCC1C2CCCCC2')))
        self.assertTrue(val.is_false_positive((['3a'], 'C1CC*CC1C2CCCCC2')))
        self.assertFalse(val.is_false_positive((['3a'], 'C1CCCCC1C2CCCCC2')))
