from pgmpy.models import JunctionTree
from pgmpy.factors import Factor
from pgmpy.tests import help_functions as hf
import numpy as np
import unittest


class TestJunctionTreeCreation(unittest.TestCase):
    def setUp(self):
        self.graph = JunctionTree()

    def test_add_single_node(self):
        self.graph.add_node(('a', 'b'))
        self.assertListEqual(self.graph.nodes(), [('a', 'b')])

    def test_add_single_node_raises_error(self):
        self.assertRaises(TypeError, self.graph.add_node, 'a')

    def test_add_multiple_nodes(self):
        self.graph.add_nodes_from([('a', 'b'), ('b', 'c')])
        self.assertListEqual(hf.recursive_sorted(self.graph.nodes()),
                             [['a', 'b'], ['b', 'c']])

    def test_add_single_edge(self):
        self.graph.add_edge(('a', 'b'), ('b', 'c'))
        self.assertListEqual(hf.recursive_sorted(self.graph.nodes()),
                             [['a', 'b'], ['b', 'c']])
        self.assertListEqual(sorted([node for edge in self.graph.edges()
                                     for node in edge]),
                             [('a', 'b'), ('b', 'c')])

    def test_add_single_edge_raises_error(self):
        self.assertRaises(ValueError, self.graph.add_edge,
                          ('a', 'b'), ('c', 'd'))

    def test_add_cyclic_path_raises_error(self):
        self.graph.add_edge(('a', 'b'), ('b', 'c'))
        self.graph.add_edge(('b', 'c'), ('c', 'd'))
        self.assertRaises(ValueError, self.graph.add_edge, ('c', 'd'), ('a', 'b'))

    def tearDown(self):
        del self.graph

class TestJunctionTreeCopy(unittest.TestCase):
    def setUp(self):
        self.graph = JunctionTree()

    def test_copy(self):
        self.graph.add_nodes_from([('a', 'b', 'c'), ('a', 'b') ,('a', 'c')])
        self.graph.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
                                   (('a', 'b', 'c'), ('a', 'c'))])
        graph_copy = self.graph.copy()
        self.graph.remove_edge(('a', 'b', 'c'), ('a', 'c'))
        self.assertFalse(self.graph.has_edge(('a', 'b', 'c'), ('a', 'c')))
        self.assertTrue(graph_copy.has_edge(('a', 'b', 'c'), ('a', 'c')))
        self.graph.remove_node(('a', 'c'))
        self.assertFalse(self.graph.has_node(('a', 'c')))
        self.assertTrue(graph_copy.has_node(('a', 'c')))
        self.graph.add_node(('c', 'd'))
        self.assertTrue(self.graph.has_node(('c', 'd')))
        self.assertFalse(graph_copy.has_node(('c', 'd')))

    def test_copy1(self):
        self.graph.add_edges_from([[('a', 'b'), ('b', 'c')]])
        phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        graph_copy = self.graph.copy()
        self.assertIsInstance(graph_copy, JunctionTree)
        self.assertIsNot(self.graph, graph_copy)
        self.assertEqual(hf.recursive_sorted(self.graph.nodes()), 
                         hf.recursive_sorted(graph_copy.nodes()))
        self.assertEqual(hf.recursive_sorted(self.graph.edges()), 
                         hf.recursive_sorted(graph_copy.edges()))
        self.assertTrue(graph_copy.check_model())
        self.assertEqual(self.graph.get_factors(), graph_copy.get_factors())
        self.graph.remove_factors(phi1, phi2)
        self.assertTrue(phi1 not in self.graph.factors and phi2 not in self.graph.factors)
        self.assertTrue(phi1 in graph_copy.factors and phi2 in graph_copy.factors)

    def tearDown(self):
        del self.graph
