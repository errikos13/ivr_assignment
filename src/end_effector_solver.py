#!/usr/bin/env python


import numpy as np
from scipy.optimize import least_squares

from fk import robot


def solve_joint_angles(end_effector_pos, previous_angles=np.zeros(4)):
    """ Return joint angles needed to put end effector at end_effector_pos. end_effector_pos is 3x1 array. """
    minimising_function = lambda joint_angles: np.linalg.norm(robot.K(joint_angles) - end_effector_pos)
    return least_squares(minimising_function, previous_angles, bounds=([-np.pi, -np.pi/2, -np.pi/2, -np.pi/2], [np.pi, np.pi/2, np.pi/2, np.pi/2])).x


def jacobian(q):
    """ a, b, c, d are the thetas for each link """
    a, b, c, d = q
    a -= np.pi/2
    b -= np.pi/2
    return np.matrix([
        [-3*np.cos(b)*np.cos(c)*np.sin(a)-3*np.cos(a)*np.sin(c)+2*np.cos(d)*(-np.cos(b)*np.cos(c)*np.sin(a)-np.cos(a)*np.sin(c))+2*np.sin(a)*np.sin(b)*np.sin(d),
            -3*np.cos(a)*np.cos(c)*np.sin(b)-2*np.cos(a)*np.cos(c)*np.cos(d)*np.sin(b)-2*np.cos(a)*np.cos(b)*np.sin(d),
            -3*np.cos(c)*np.sin(a)-3*np.cos(a)*np.cos(b)*np.sin(c)+2*np.cos(d)*(-np.cos(c)*np.sin(a)-np.cos(a)*np.cos(b)*np.sin(c)),
            -2*np.cos(a)*np.cos(d)*np.sin(b)-2*(np.cos(a)*np.cos(b)*np.cos(c)-np.sin(a)*np.sin(c))*np.sin(d)],
        [3*np.cos(a)*np.cos(b)*np.cos(c)-3*np.sin(a)*np.sin(c)+2*np.cos(d)*(np.cos(a)*np.cos(b)*np.cos(c)-np.sin(a)*np.sin(c))-2*np.cos(a)*np.sin(b)*np.sin(d),
            -3*np.cos(c)*np.sin(a)*np.sin(b)-2*np.cos(c)*np.cos(d)*np.sin(a)*np.sin(b)-2*np.cos(b)*np.sin(a)*np.sin(d),
            3*np.cos(a)*np.cos(c)-3*np.cos(b)*np.sin(a)*np.sin(c)+2*np.cos(d)*(np.cos(a)*np.cos(c)-np.cos(b)*np.sin(a)*np.sin(c)),
            -2*np.cos(d)*np.sin(a)*np.sin(b)-2*(np.cos(b)*np.cos(c)*np.sin(a)+np.cos(a)*np.sin(c))*np.sin(d)],
        [0,
            -3*np.cos(b)*np.cos(c)-2*np.cos(b)*np.cos(c)*np.cos(d)+2*np.sin(b)*np.sin(d),
            3*np.sin(b)*np.sin(c)+2*np.cos(d)*np.sin(b)*np.sin(c),
            -2*np.cos(b)*np.cos(d)+2*np.cos(c)*np.sin(b)*np.sin(d)]])
        #  [0,0,1,0],
        #  [0,1,0,1],
        #  [1,0,0,0]])
