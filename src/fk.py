#!/usr/bin/env python


from functools import reduce

import numpy as np


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
    return rotZ(theta).dot(translate(r, 0, d)).dot(tX(alpha))


def robot_matrix(d, theta, r, alpha):
    """ Inputs are n-element arrays (same n) containing the joint parameters. """
    return reduce(np.matmul, [ link_matrix(d[i], theta[i], r[i], alpha[i]) for i in range(len(d)) ])


class Robot:
    origin = np.array([0, 0, 0, 1])

    def __init__(self, d, theta, r, alpha):
        """ Inputs are n-element arrays (same n) containing the joint parameters of the robot when it is unrotated. """
        self.d = d
        self.theta = theta
        self.r = r
        self.alpha = alpha

    def K(self, joint_angles):
        """ Return the 3x1 vector of the location of the end effector for input joint angles. """
        theta = [ self.theta[i] + joint_angles[i] for i in range(len(joint_angles)) ]
        return robot_matrix(self.d, theta, self.r, self.alpha).dot(self.origin).A1[:3]


""" Our robot. """
robot = Robot([2, 0, 0, 0], [-np.pi/2, -np.pi/2, 0, 0], [0, 0, 3, 2], [-np.pi/2, np.pi/2, -np.pi/2, 0])
