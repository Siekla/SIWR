#!/usr/bin/env python

"""code template"""

import numpy as np
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor


def zad_6():
    G = FactorGraph()
    G.add_nodes_from(['r0', 'r1', 'r2', 'r3', 'r4', 'u0', 'u1', 'u2', 'u3', 'u4'])

    phi0 = DiscreteFactor(['r0', 'r1'], [2, 1], [0.6, 0.4])
    phi1 = DiscreteFactor(['r1', 'r2'], [2, 2], [[0.7, 0.3], [0.3, 0.7]])
    phi2 = DiscreteFactor(['r2', 'r3'], [2, 2], [[0.7, 0.3], [0.3, 0.7]])
    phi3 = DiscreteFactor(['r3', 'r4'], [2, 2], [[0.7, 0.3], [0.3, 0.7]])
    phi4 = DiscreteFactor(['r0', 'u0'], [2, 2], [[0.9, 0.1], [0.2, 0.8]])
    phi5 = DiscreteFactor(['r1', 'u1'], [2, 2], [[0.9, 0.1], [0.2, 0.8]])
    phi6 = DiscreteFactor(['r2', 'u2'], [2, 2], [[0.9, 0.1], [0.2, 0.8]])
    phi7 = DiscreteFactor(['r3', 'u3'], [2, 2], [[0.9, 0.1], [0.2, 0.8]])
    phi8 = DiscreteFactor(['r4', 'u4'], [2, 2], [[0.9, 0.1], [0.2, 0.8]])

    G.add_factors(phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8)

    G.add_edges_from([('r0', phi1), ('r1', phi1),
                      ('r1', phi1), ('r2', phi1),
                      ('r2', phi1), ('r3', phi1),
                      ('r3', phi1), ('r4', phi1),
                      ('r0', phi1), ('u0', phi1),
                      ('r1', phi1), ('u1', phi1),
                      ('r2', phi1), ('u2', phi1),
                      ('r3', phi1), ('u3', phi1),
                      ('r4', phi1), ('u4', phi1)])

    G.get_variable_nodes()

    q0 = bel.query(veriable=['R0'])
    q1 = bel.query(veriable=['R0'], evidence={'U1':0})
    q2 = bel.query(veriable=['R0'], evidence={'U2': 0, 'U1':0})
    q3 = bel.query(veriable=['R0'], evidence={'U3': 0, 'U2':0, 'U1':1})


if __name__ == '__main__':
    zad_6()