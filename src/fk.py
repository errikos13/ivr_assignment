#!/usr/bin/env python

import numpy as np

last_link_length = 2

# all coordinates are in homogenous coordinates

# TODO
def get_dh_parameters(rotations):
    # for no rotations in robot
    dh_parameters = [
            {
                "d": 2,
                "theta": 0,
                "r": 0,
                "alpha": -np.pi/2,
            },
            {
                "d": 0,
                "theta": -np.pi/2,
                "r": 0,
                "alpha": -np.pi/2,
            },
            {
                "d": 0,
                "theta": 0,
                "r": 3,
                "alpha": np.pi/2,
            },
        ]

def translate(dx, dy, dz):
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

# link 0 is the yellow link (called link 1 in coursework document)
def dh_transform(point, parameters):
    for p in parameters:
        point = rotX(p["alpha"]).dot(point)
        point = translate(p["r"], 0, p["d"]).dot(point)
        point = rotZ(p["theta"]).dot(point)
    return point

