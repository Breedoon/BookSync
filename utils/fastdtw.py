#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division
import numbers
import numpy as np
from collections import defaultdict
from pyts.metrics import itakura_parallelogram, sakoe_chiba_band


def fastdtw(x, y, radius=1, dist=None, max_approximations=0):
    ''' return the approximate distance between 2 time series with O(N)
        time and memory complexity

        Parameters
        ----------
        x : array_like
            input array 1
        y : array_like
            input array 2
        radius : int
            size of neighborhood when expanding the path. A higher value will
            increase the accuracy of the calculation but also increase time
            and memory consumption. A radius equal to the size of x and y will
            yield an exact dynamic time warping calculation.
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.
        max_approximations : int
            The maximum number of recursive iterations for path band approximations
            that fastdtw will go through before running the actual final algorithm.
            On each iteration, fastdtw will halve both arrays and recursively run
            itself to find path in the shortened array.

        Returns
        -------
        distance : float
            the approximate distance between the 2 time series
        path : list
            list of indexes for the inputs x and y

        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw
        >>> x = np.array([1, 2, 3, 4, 5], dtype='float')
        >>> y = np.array([2, 3, 4], dtype='float')
        >>> fastdtw.fastdtw(x, y)
        (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])
    '''
    x, y, dist = __prep_inputs(x, y, dist)
    return __fastdtw(x, y, radius, dist, max_approximations)


def __difference(a, b):
    return abs(a - b)


def __norm(p):
    return lambda a, b: np.linalg.norm(np.atleast_1d(a) - np.atleast_1d(b), p)


def __fastdtw(x, y, radius, dist, max_approximations, curr_approximation=0):
    if curr_approximation >= max_approximations:
        return dtw(x, y, radius, dist=dist)

    x_shrinked = __reduce_by_half(x)
    y_shrinked = __reduce_by_half(y)
    distance, path = \
        __fastdtw(x_shrinked, y_shrinked, radius, dist, max_approximations, curr_approximation + 1)
    window = __expand_window(path, len(x), len(y), radius)
    return __dtw(x, y, window, dist=dist)


def __prep_inputs(x, y, dist):
    x = np.asanyarray(x, dtype='int')  # as int b/c assuming passed indices
    y = np.asanyarray(y, dtype='int')

    if x.ndim == y.ndim > 1 and x.shape[1] != y.shape[1]:
        raise ValueError('second dimension of x and y must be the same')
    if isinstance(dist, numbers.Number) and dist <= 0:
        raise ValueError('dist cannot be a negative integer')

    if dist is None:
        if x.ndim == 1:
            dist = __difference
        else:
            dist = __norm(p=1)
    elif isinstance(dist, numbers.Number):
        dist = __norm(p=dist)

    return x, y, dist


def dtw(x, y, radius=None, dist=None):
    ''' return the distance between 2 time series without approximation

        Parameters
        ----------
        x : array_like
            input array 1
        y : array_like
            input array 2
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.

        Returns
        -------
        distance : float
            the approximate distance between the 2 time series
        path : list
            list of indexes for the inputs x and y

        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw
        >>> x = np.array([1, 2, 3, 4, 5], dtype='float')
        >>> y = np.array([2, 3, 4], dtype='float')
        >>> fastdtw.dtw(x, y)
        (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])
    '''
    x, y, dist = __prep_inputs(x, y, dist)
    return __dtw(x, y, radius, dist)


def __dtw(x, y, window, dist):
    from tqdm import tqdm

    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]

    D = defaultdict(lambda: (float('inf'),))
    D[0, 0] = (0, 0, 0)

    if hasattr(window, '__iter__'):  # a list of tuples
        window = ((i + 1, j + 1) for i, j in window)
        for i, j in tqdm(list(window)):
            dt = dist(x[i - 1], y[j - 1])
            D[i, j] = min((D[i - 1, j][0] + dt, i - 1, j),
                          # (D[i, j - 1][0] + dt, i, j - 1),
                          (D[i - 1, j - 1][0] + dt, i - 1, j - 1),
                          key=lambda a: a[0])

    elif isinstance(window, int):  # given a radius
        js = sakoe_chiba_band(len(x), len(y), window * 2).T + 1
        for i in tqdm(list(range(1, len(x) + 1))):
            for j in range(*js[i - 1]):
                dt = dist(x[i - 1], y[j - 1])
                D[i, j] = min((D[i - 1, j][0] + dt, i - 1, j),
                              # (D[i, j - 1][0] + dt, i, j - 1),
                              (D[i - 1, j - 1][0] + dt, i - 1, j - 1),
                              key=lambda a: a[0])

    path = []
    i, j = len_x, len_y
    while not (i == j == 0):
        path.append((i - 1, j - 1))
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()
    return (D[len_x, len_y][0], path)


def __reduce_by_half(x):
    thickness = x.shape[1]
    oddness = len(x) % 2
    # Pad with -1 at the end if len is odd
    x = np.vstack([x, np.array([-1] * thickness * oddness, dtype=int).reshape(oddness, thickness)])
    return x.reshape(-1, thickness * 2)


def __expand_window(path, len_x, len_y, radius):
    path_arr = np.array(path)

    short_windows = np.repeat(path_arr[:, 1, np.newaxis] + np.arange(-radius, radius + 1)[np.newaxis, :],
                              2, axis=0)[:(len(path_arr) * 2) - len_x % 2].flatten()  # repeat for every pair of x

    expanded_windows = np.array([
        np.repeat(np.arange(len_x), 2 * (radius * 2 + 1)),  # inds of x: 0 0 0 0 0 1 1 1 1
        np.vstack([short_windows * 2, (short_windows * 2) + 1]).T.flatten()  # inds if y
    ]).T

    # remove ones with incorrect indices
    windows = expanded_windows[(expanded_windows >= 0).all(axis=1) & (expanded_windows[:, 1] < len_y)]

    return windows
