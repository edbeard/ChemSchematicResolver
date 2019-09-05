# -*- coding: utf-8 -*-
"""
Image IO
========
Reading and writing images.

Module adapted by :-
author: Ed Beard
email: ejb207@cam.ac.uk

from FigureDataExtractor (<CITATION>) :-
author: Matthew Swain
email: m.swain@me.com

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import numpy as np
from PIL import Image
from skimage import img_as_float, img_as_ubyte, img_as_uint
from skimage import io as skio
from skimage.color import gray2rgb
import os

from .model import Figure

log = logging.getLogger(__name__)


def imread(f, raw=False):
    """Read an image from a file, create Figure object
    :param string|file f: Filename or file-like object.
    :return: Figure object.
    :rtype: Figure
    """

    img = skio.imread(f, plugin='pil')

    # Transform all images pixel values to be floating point values between 0 and 1 (i.e. not ints 0-255)
    # Recommended in skimage-tutorials "Images are numpy arrays" because this what scikit-image uses internally
    # Transform greyscale images to RGB
    if len(img.shape) == 2:
        log.debug('Converting greyscale image to RGB...')
        img = gray2rgb(img)

    # Transform all images pixel values to be floating point values between 0 and 1 (i.e. not ints 0-255)
    # Recommended in skimage-tutorials "Images are numpy arrays" because this what scikit-image uses internally
    if not raw:
        img = img_as_float(img)
    fig = Figure(img)

    return fig


def imsave(f, img):
    """Save an image to file.
    :param string|file f: Filename or file-like object.
    :param numpy.ndarray img: Image to save. Of shape (M,N) or (M,N,3) or (M,N,4).
    """
    # Ensure we use PIL so we can guarantee that imsave will accept file-like object as well as filename
    skio.imsave(f, img, plugin='pil', quality=100)


def imdel(f):
    """ Delete an image file
    """

    os.remove(f)


def img_as_pil(arr, format_str=None):
    """Convert an scikit-image image (ndarray) to a PIL object.

    Derived from code in scikit-image PIL IO plugin.

    :param numpy.ndarray img: The image to convert.
    :return: PIL image.
    :rtype: Image
    """
    if arr.ndim == 3:
        arr = img_as_ubyte(arr)
        mode = {3: 'RGB', 4: 'RGBA'}[arr.shape[2]]

    elif format_str in ['png', 'PNG']:
        mode = 'I;16'
        mode_base = 'I'

        if arr.dtype.kind == 'f':
            arr = img_as_uint(arr)

        elif arr.max() < 256 and arr.min() >= 0:
            arr = arr.astype(np.uint8)
            mode = mode_base = 'L'

        else:
            arr = img_as_uint(arr)

    else:
        arr = img_as_ubyte(arr)
        mode = 'L'
        mode_base = 'L'

    try:
        array_buffer = arr.tobytes()
    except AttributeError:
        array_buffer = arr.tostring()  # Numpy < 1.9

    if arr.ndim == 2:
        im = Image.new(mode_base, arr.T.shape)
        try:
            im.frombytes(array_buffer, 'raw', mode)
        except AttributeError:
            im.fromstring(array_buffer, 'raw', mode)  # PIL 1.1.7
    else:
        image_shape = (arr.shape[1], arr.shape[0])
        try:
            im = Image.frombytes(mode, image_shape, array_buffer)
        except AttributeError:
            im = Image.fromstring(mode, image_shape, array_buffer)  # PIL 1.1.7
    return im
