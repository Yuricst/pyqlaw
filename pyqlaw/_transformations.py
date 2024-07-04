"""
Transformation functions
"""

from numba import njit
import numpy as np

@njit
def rotmat1(phi):
	return np.array([ [1.0, 0.0, 0.0],
					  [0.0, np.cos(phi), np.sin(phi)],
					  [0.0, -np.sin(phi), np.cos(phi)] ])

@njit
def rotmat2(phi):
	return np.array([ [np.cos(phi), 0.0, -np.sin(phi)],
					  [0.0, 1.0, 0.0],
					  [np.sin(phi), 0.0, np.cos(phi)] ])

@njit
def rotmat3(phi):
	return np.array([ [ np.cos(phi), np.sin(phi), 0.0],
					  [-np.sin(phi), np.cos(phi), 0.0],
					  [0.0, 0.0, 1.0] ])


def wrap(alpha):
	"""Wrap angle to be between -pi and pi"""
	return (alpha + np.pi) % 2.0 * np.pi - np.pi