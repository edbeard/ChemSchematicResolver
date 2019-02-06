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

import chemschematicdiagramextractor as csde
import unittest
import numpy as np

class TestValidation(unittest.TestCase):

    # def test_get_pixel_ratio(self):
    #     # 16 black, 48 white
    #     img = np.ones((4, 4))
    #     img = np.pad(img, 2, 'constant')

    #     ratio = csde.validate.get_pixel_ratio(img)
    #     self.assertEqual(ratio, 0.25)

    def test_pybel_checks(self):
        self.assertEqual(csde.validate.pybel_format("&&"), None)
        self.assertEqual(csde.validate.pybel_format("C1CCCCC1C2CCCCC2"), 'C1CCC(CC1)C1CCCCC1')
