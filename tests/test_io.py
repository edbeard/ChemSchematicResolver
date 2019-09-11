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

import chemschematicresolver as csr
import os
from pathlib import Path
import unittest

log = logging.getLogger(__name__)

tests_dir = os.path.abspath(__file__)
data_dir = os.path.join(os.path.dirname(tests_dir), 'data')
sample_diag = os.path.join(data_dir, 'S014372081630122X_gr1.jpg')


class TestImportAndSave(unittest.TestCase):
    """ Tests importing and saving of relevant image types."""

    def test_import_jpg(self):
        """ Tests import of jpg file"""

        fig = csr.io.imread(sample_diag)

        output_path = os.path.join(data_dir, 'test_import_and_save.jpg')
        csr.io.imsave(output_path,fig.img)
        f = Path(output_path)
        is_file = f.is_file()
        os.remove(output_path)

        self.assertTrue(is_file)



