#!/usr/bin/env python

from functools import reduce

import numpy as np

# all coordinates are in homogenous coordinates

""" The joint parameters for our robot, in its initial unrotated state. When the robot rotates only the thetas will change. """
d = [2, 0, 0, 0]
theta = [-np.pi/2, -np.pi/2, 0, 0]
r = [0, 0, 3, 2]
alpha = [-np.pi/2, np.pi/2, -np.pi/2, 0]

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

def link_matrix(d, theta, r, alpha):
    return rotZ(theta) @ translate(r, 0, d) @ rotX(alpha)

def robot_matrix(d, theta, r, alpha):
    """ Inputs are n-element arrays (same n) containing the joint parameters. """
    return reduce(np.matmul, [ link_matrix(d[i], theta[i], r[i], alpha[i]) for i in range(len(d)) ])

