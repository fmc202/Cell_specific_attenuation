# --------------------------------------------------------------------------------------------------------------------------------------------------
# Author: Chenying Liu
# Created Date: Dec-26-2020
# Last Modified: Dec-26-2020
# Copyright (c) 2020
# Description: generate the R matrix for cell-specific attenuation given a number of points and cells
# Features: Using JIT with LLVM compiler and fastmath with intel SVML to achieve 500x faster and 5000x faster than pure python and R codes
#           The static types are specified for JIT compliation before runtime
# Warning: always use numpy array with float64 for performance
# Usage: install numpy and numba and import get_fraction function from this  module
# --------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import numba
from numba import jit, int64, double


@jit(
    int64(double, double, double, double, double, double),
    nopython=True,
    fastmath=True,
)
def compute_code(x, y, x_min, y_min, x_max, y_max):
    """called internally by clip_cell function

    Args:
        x (double): x coordinate of one endpoint of the path
        y (double): y coordinate of one endpoint of the path
        x_min (double): min x coordinate of the cell
        y_min (double): min y of the cell
        x_max (double): max x of the cell
        y_max (double): max y of the cell

    Returns:
        int: region code of point(x,y)
    """

    inside = 0
    left = 1
    right = 2
    bottom = 4
    top = 8
    code = inside

    if x < x_min:
        code = code | left
    elif x > x_max:
        code = code | right

    if y < y_min:
        code = code | bottom
    elif y > y_max:
        code = code | top

    return code


@jit(
    double(double, double, double, double, double, double, double, double),
    nopython=True,
    fastmath=True,
)
def clip_cell(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
    """get the fraction of length in a cell given the two endpoints of a path

    Args:
        x1 (double): x coord of point 1
        y1 (double): y coord of point 1
        x2 (double): x coord of point 2
        y2 (double): y coord of point 2
        x_min (double): cell coord defined as in compute code
        y_min (double): cell coord defined as in compute code
        x_max (double): cell coord defined as in compute code
        y_max (double): cell coord defined as in compute code

    Returns:
        double: length in the cell in fraction
    """
    left = 1
    right = 2
    bottom = 4
    top = 8

    code1 = compute_code(x1, y1, x_min, y_min, x_max, y_max)
    code2 = compute_code(x2, y2, x_min, y_min, x_max, y_max)
    length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    while True:
        if code1 == 0 and code2 == 0:
            R = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / length
            break
        elif (code1 & code2) != 0:
            R = 0
            break
        else:
            if code1 != 0:
                code_out = code1
            else:
                code_out = code2

            if code_out & top:
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
                y = y_max
            elif code_out & bottom:
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
                y = y_min
            elif code_out & right:
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
                x = x_max
            elif code_out & left:
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
                x = x_min

            if code_out == code1:
                x1 = x
                y1 = y
                code1 = compute_code(x1, y1, x_min, y_min, x_max, y_max)
            else:
                x2 = x
                y2 = y
                code2 = compute_code(x2, y2, x_min, y_min, x_max, y_max)

    return R


@jit(
    numba.types.Tuple((double[:, :], double[:, :]))(double[:], double[:], double[:], double[:], double[:], double[:], double[:]),
    nopython=True,
    fastmath=True,
)
def get_fraction(x, y, X1, Y1, X2, Y2, R):
    """compute length in all cells for all data

    Args:
        x (1d-array): spacing of x in a meshgrid (e.g x is the argument in np.meshgrid(x,y))
        y (1d-array): spacing of y in a meshgrid
        X1 (1d-array): x coord of all point 1
        Y1 (1d-array): y coord of all point 1
        X2 (1d-array): x coord of all point 2
        Y2 (1d-array): y coord of all point 2
        R (1d-array): a vector of rupture distance

    Returns:
        2d-array: a matrix of shape (no_data,no_cell)\n
         with each entry representing the length in the cell (e.g. km)
    """
    no_x = x.shape[0] - 1
    no_y = y.shape[0] - 1
    no_cell = no_x * no_y
    no_data = X1.shape[0]

    # use np.repeat instead of tile for numba compatibility
    x_min = np.repeat(x[:-1], no_y).reshape(-1, no_y).T.flatten()  # x_min = np.tile(x[:-1], no_y)
    x_max = np.repeat(x[1:], no_y).reshape(-1, no_y).T.flatten()  # x_max = np.tile(x[1:], no_y)
    y_min = np.repeat(y[:-1], no_x)  # y_min = np.tile(y[:-1], no_x)
    y_max = np.repeat(y[1:], no_x)  # y_max = np.tile(y[1:], no_x)

    # try to use tuple instead of list in np.empty() to prevent type changing for numba compatibility
    R_frac = np.empty((no_data, no_cell))
    mid_coord = np.empty((no_cell, 2))

    for i in range(no_data):
        for j in range(no_cell):
            R_frac[i, j] = R[i] * clip_cell(X1[i], Y1[i], X2[i], Y2[i], x_min[j], y_min[j], x_max[j], y_max[j])

    for j in range(no_cell):
        mid_coord[j, 0] = (x_min[j] + x_max[j]) / 2
        mid_coord[j, 1] = (y_min[j] + y_max[j]) / 2

    return mid_coord, R_frac
