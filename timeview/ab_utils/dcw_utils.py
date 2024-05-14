"""
Created on 10/05/2024
@author: Antonin Berthon

Utilities for Dynamic Composition Warping.
"""
import numpy as np


def prepare_compositions(s1, t1, s2, t2):
    """
    Extend two compositions so they share common transition points.
    """
    # Find the all transition points
    common_t = sorted(set(t1) | set(t2))

    # Extend the compositions to the common transition points
    s1_extended = extend_composition(s1, t1, common_t)
    s2_extended = extend_composition(s2, t2, common_t)
    return s1_extended, s2_extended, common_t


def extend_composition(s, t, common_t):
    """
    Extend a composition to the common transition points.
    Example:
    Input:
    - (s, t) = ([1, 2, 3], [0, 0.1, 0.3, 1])
    - common_t = [0., 0.05, 0.1, 0.3, 0.5, 1]
    Output: [1, 1, 2, 3, 3], [0, 0.05, 0.1, 0.3, 0.5, 1]
    """
    assert all([t_i in common_t for t_i in t])  # t should be included in common_t

    s_extended = []
    idx = 0
    for ct in common_t[:-1]:
        while ct >= t[idx + 1] and idx < len(t) - 2:
            idx += 1
        s_extended.append(s[idx])
    return s_extended


def d_intervals(interval1, interval2):
    """Computes the distance between two intervals by representing each as a point in 2D space
    and calculating the Euclidean distance between them."""
    return np.sqrt(
        (interval1[0] - interval2[0]) ** 2 + (interval1[1] - interval2[1]) ** 2
    )


def get_weight_time_func():
    return lambda a, b: d_intervals(a, b) / np.sqrt(2)


def compute_dcw(
    s1, t1, s2, t2, distance_func=None, weight_func=None, penalty_func=None, lamb=0
):
    n, m = len(s1), len(s2)
    # Initialize the DTW matrix with infinite values
    dtw_matrix = np.full((n, m), float("inf"))

    # Define a default weight function if none provided
    if (
        weight_func is None
    ):  # weight a particular match: a bad match on small segment is less important than on a large segment
        weight_func = lambda a, b: 1  # Uniform weight by default

    if penalty_func is None:  # penalty for time shift
        penalty_func = get_weight_time_func()

    if distance_func is None:
        distance_func = lambda a, b: 1 - int(a == b)

    # Populate the DTW matrix
    for i in range(0, n):
        for j in range(0, m):  # fill columns by columns
            # distance between the two elements
            dist_ij = distance_func(s1[i], s2[j])
            # weight according to time
            interval1, interval2 = [t1[i], t1[i + 1]], [t2[j], t2[j + 1]]
            weight_ij = weight_func(interval1, interval2)
            # time penalty term
            penalty_func_ij = penalty_func(interval1, interval2)
            val = weight_ij * dist_ij + lamb * penalty_func_ij

            if i == 0 and j == 0:  # init first cell
                dtw_matrix[i, j] = val
                continue

            bottom = dtw_matrix[i - 1, j] if i > 0 else float("inf")
            left = dtw_matrix[i, j - 1] if j > 0 else float("inf")
            diag = dtw_matrix[i - 1, j - 1] if i > 0 and j > 0 else float("inf")
            min_cost = min(bottom, left, diag)
            dtw_matrix[i, j] = val + min_cost
    return dtw_matrix[-1, -1]


# Useful distances for dcw
def interval_euclidean_distance():
    return lambda a, b: d_intervals(a, b) / np.sqrt(2)


def average_interval_size():
    return lambda a, b: (a[1] - a[0] + b[1] - b[0]) / 2


def min_interval_size():
    return lambda a, b: min(a[1] - a[0], b[1] - b[0])
