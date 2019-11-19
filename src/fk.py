#!/usr/bin/env python

import numpy as np

# all coordinates are in homogenous coordinates

def translate(dx, dy, dz):
    """ Return the transformation matrix for the translation through (dx, dy, dz)T """
    return np.matrix([
        [ 1, 0, 0, dx ],
        [ 0, 1, 0, dy ],
        [ 0, 0, 1, dz ],
        [ 0, 0, 0, 1 ]])

def rotZ(theta):
    return np.matrix([
        [  np.cos(theta), np.sin(theta), 0, 0 ],
        [ -np.sin(theta), np.cos(theta), 0, 0 ],
        [  0,             0,             1, 0 ],
        [  0,             0,             0, 1 ]])

def rotX(alpha):
    return np.matrix([
        [ 1,  0,             0,             0 ],
        [ 0,  np.cos(alpha), np.sin(alpha), 0 ],
        [ 0, -np.sin(alpha), np.cos(alpha), 0 ],
        [ 0,  0,             0,             1 ]])

def dh_transformation(d, theta, r, alpha):
    return rotZ(theta) @ translate(r, 0, d) @ rotX(alpha)

