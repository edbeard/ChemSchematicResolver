# -*- coding: utf-8 -*-
"""
test_model
========

Test model functionality

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

log = logging.getLogger(__name__)

import chemschematicresolver.model as mod
import unittest

class TestModel(unittest.TestCase):

    def test_graph_kruskal(self):
        g = mod.Graph(4)
        g.addEdge(0, 1, 10)
        g.addEdge(0, 2, 6)
        g.addEdge(0, 3, 5)
        g.addEdge(1, 3, 15)
        g.addEdge(2, 3, 4)

        result = g.kruskal()

        self.assertEqual(result, [[2, 3, 4], [0, 3, 5], [0, 1, 10]])

    def test_separation(self):
        r1 = mod.Rect(-1, 1, -1, 1)
        r2 = mod.Rect(2, 4, 3, 5)
        self.assertEqual(r1.separation(r2), 5.)

    def test_panel_equality(self):
        p1 = mod.Panel(1, 2, 3, 4, 0)
        p2 = mod.Panel(1, 2, 3, 4, 0)
        self.assertEqual(p1, p2)

    def test_pairs_of_panels(self):
        tuple1 = (mod.Panel(1, 2, 3, 4, 0), mod.Panel(5, 6, 7, 8, 0))
        tuple2 = (mod.Panel(5, 6, 7, 8, 0), mod.Panel(1, 2, 3, 4, 0))

        list1 = [tuple1, tuple2]
        list2 = [tuple1, tuple2]

        if tuple1 in list1:
                print('foo')
