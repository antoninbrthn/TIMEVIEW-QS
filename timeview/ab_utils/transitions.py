"""
Created on 09/05/2024
@author: Antonin Berthon

Utilities for mutation matrices.
Currently expects to be using the dynamical motifs as defined in the STATES variable in basis.py
"""
import numpy as np

from timeview.basis import BSplineBasis

N_STATES = len(BSplineBasis.STATES)  # dynamical motifs
T_ID = 1 - np.eye(N_STATES)


def get_binary_trend_matrix():
    """Mutation matrix encoding binary trends: either increasing or decreasing."""
    mutation_matrix = np.ones((7, 7))  # init as all ones
    increasing_indices = [
        i for i, state in enumerate(BSplineBasis.STATES) if "increasing" in state
    ]
    for i in increasing_indices:
        for j in increasing_indices:
            mutation_matrix[i, j] = 0
    decreasing_indices = [
        i for i, state in enumerate(BSplineBasis.STATES) if "decreasing" in state
    ]
    for i in decreasing_indices:
        for j in decreasing_indices:
            mutation_matrix[i, j] = 0
    return mutation_matrix


T_BINARY_TREND = get_binary_trend_matrix()
