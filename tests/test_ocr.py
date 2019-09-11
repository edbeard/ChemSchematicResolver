# -*- coding: utf-8 -*-
"""
test_ocr
========

Test optical character recognition
TODO : Choose a working OCR example... OR add neural net to improve
"""

import unittest
import os
import chemschematicresolver as csr
import copy

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

tests_dir = os.path.abspath(__file__)
test_ocr_dir = os.path.join(os.path.dirname(tests_dir), 'data', 'ocr')


class TestOcr(unittest.TestCase):

    def test_ocr_all_imgs(self):
        """
        Uses the OCR module on the whole image to identify text blocks
        """
        test_imgs = [os.path.join(test_ocr_dir, file) for file in os.listdir(test_ocr_dir)]

        for img_path in test_imgs:
            fig = csr.io.imread(img_path)  # Read in float and raw pixel images
            text_blocks = csr.ocr.get_text(fig.img)

            # Create output image
            out_fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(fig.img)

        self.assert_equal(text_blocks[0].text, '1: R1=R2=H:TQEN\n2:R1=H,R2=OMe:T(MQ)EN\n3: R1=R2=OMe:T(TMQ)EN\n\n')

    def test_ocr_r_group(self):
        """
        Used to test different functions on OCR recognition"""

        path = os.path.join(test_ocr_dir, 'S0143720816301115_gr1_text.jpg')

        fig = csr.io.imread(path)
        copy_fig = copy.deepcopy(fig)

        bin_fig = copy_fig

        text_blocks = csr.ocr.get_text(bin_fig.img)

        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(bin_fig.img)
        plt.show()

        print(text_blocks)






