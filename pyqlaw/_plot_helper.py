"""
Helper functions for plotting
"""

import numpy as np
import matplotlib.pyplot as plt

def get_sphere_coordinates(radius, center=None):
    """Get x,y,z coordinates for sphere

    Args:
        radius (float): sphere radius
        center (list): x,y,z coordinates of center; if None, set to [0.0, 0.0, 0.0]
    
    Returns:
        (tuple): x, y, z coordinates of sphere
    """
    # check if center is provided
    if center is None:
        center = [0.0, 0.0, 0.0]
    # construct reference sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    #u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sphere = center[0] + radius*np.cos(u)*np.sin(v)
    y_sphere = center[1] + radius*np.sin(u)*np.sin(v)
    z_sphere = center[2] + radius*np.cos(v)
    return x_sphere, y_sphere, z_sphere


def plot_sphere_wireframe(ax, radius, center=None, color="k", linewidth=0.5):
    """Plot sphere wireframe
    
    Args:
        ax (Axes3DSubplot): matplotlib 3D axis, created by `ax = fig.add_subplot(projection='3d')`
        radius (float): radius
        center (list): x,y,z coordinates of center, if None set to [0.0, 0.0, 0.0]
        color (str): color of wireframe
        linewidth (float): linewidth of wireframe
    """
    x_sphere, y_sphere, z_sphere = get_sphere_coordinates(radius, center)
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color=color, linewidth=linewidth)
    return


def get_ellipsoid_coordinates(rx, ry, rz, center=None, n=60):
    """Get x,y,z coordinates for ellipsoid

    Args:
        rx (float): radius in x-direction
        ry (float): radius in y-direction
        rz (float): radius in z-direction
        center (list): x,y,z coordinates of center; if None, set to [0.0, 0.0, 0.0]
        n (int): number of points to be used in each mesh direction

    Returns:
        (tuple): x, y, z coordinates of ellipsoid
    """
    # check if center is provided
    if center is None:
        center = [0.0, 0.0, 0.0]

    # grid setup
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)

    # Cartesian coordinates of ellipsoid
    x_el = center[0] + rx * np.outer(np.cos(u), np.sin(v))
    y_el = center[1] + ry * np.outer(np.sin(u), np.sin(v))
    z_el = center[2] +rz * np.outer(np.ones_like(u), np.cos(v))
    return x_el, y_el, z_el


def plot_ellipsoid_wireframe(ax, rx, ry, rz, center=None, color="k", linewidth=0.5, n=60):
    """Plot ellipsoid wireframe

    Args:
        ax (Axes3DSubplot): matplotlib 3D axis, created by `ax = fig.add_subplot(projection='3d')`
        rx (float): radius in x-direction
        ry (float): radius in y-direction
        rz (float): radius in z-direction
        center (list): x,y,z coordinates of center, if None set to [0.0, 0.0, 0.0]
        color (str): color of wireframe
        linewidth (float): linewidth of wireframe
        n (int): number of points to be used in each mesh direction
    """
    x_el, y_el, z_el = get_ellipsoid_coordinates(rx, ry, rz, center=center, n=n)
    ax.plot_wireframe(x_el, y_el, z_el, color=color, linewidth=linewidth)
    return


def set_equal_axis(ax, xlims, ylims, zlims, scale=1.0, dim3=True):
    """Helper function to set equal axis
    
    Args:
        ax (Axes3DSubplot): matplotlib 3D axis, created by `ax = fig.add_subplot(projection='3d')`
        xlims (list): 2-element list containing min and max value of x
        ylims (list): 2-element list containing min and max value of y
        zlims (list): 2-element list containing min and max value of z
        scale (float): scaling factor along x,y,z
        dim3 (bool): whether to also set z-limits (True for 3D plots)
    """
    # compute max required range
    max_range = np.array([max(xlims)-min(xlims), max(ylims)-min(ylims), max(zlims)-min(zlims)]).max() / 2.0
    # compute mid-point along each axis
    mid_x = (max(xlims) + min(xlims)) * 0.5
    mid_y = (max(ylims) + min(ylims)) * 0.5
    mid_z = (max(zlims) + min(zlims)) * 0.5
    # set limits to axis
    if dim3==True:
        ax.set_box_aspect((max_range, max_range, max_range))
    ax.set_xlim(mid_x - max_range*scale, mid_x + max_range*scale)
    ax.set_ylim(mid_y - max_range*scale, mid_y + max_range*scale)
    if dim3==True:
        ax.set_zlim(mid_z - max_range*scale, mid_z + max_range*scale)
    return
