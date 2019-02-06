# -*- coding: utf-8 -*-
"""
test_io
========

Test io of images.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

log = logging.getLogger(__name__)

import chemschematicdiagramextractor as csde
import os
from pathlib import Path
import unittest

tests_dir = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(os.path.dirname(tests_dir), 'examples')
sample_diag = os.path.join(example_dir, 'c4tc01753f-f1.gif')

class TestImportAndSave(unittest.TestCase):
    """ Tests importing and saving of relevant image types."""

    def test_import_png(self):
        """ Tests import of png file"""

        fig = csde.io.imread(sample_diag)

        output_path = os.path.join(example_dir, 'test_import_and_save.png')
        csde.io.imsave(output_path,fig.img)
        f = Path(output_path)
        isFile = f.is_file()
        os.remove(output_path)

        self.assertTrue(isFile)



