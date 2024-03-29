import math
from math import pi

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation


class Show3D:
    """Uses matplotlib to create an animated figure.

    Example:
        >>> # Turn on interactive mode if needed
        >>> plt.ion()
        >>> # Set plot parameters and create plotting object
        >>> sh = Show3D(cmap='gray', magnif=2, fps=24)
        >>> # Provide dynamic image
        >>> # If saving as gif, include the name of the gif
        >>> sh(dynamic_img, savename=savename)

    Args:
        clim (tuple): Min and max image values to display.
        cmap (string): Matplotlib colormap.
        magnif (float): Factor for plotting the images at a larger/smaller
            size.
        fps (float): Frames per second.
    """
    def __init__(self, clim=None, cmap='gray', magnif=1, fps=24):
        self.clim = clim
        self.cmap = cmap
        self.magnif = magnif
        self.fps = fps

    def __call__(self, img, savename=None):
        """Shows animated figure.

        Args:
            img (array): NumPy array with shape (nt, ny, nx).
            savename (string): Optional. Name of gif to save.
        """
        nt, ny, nx = img.shape
        # Set figure
        f = plt.figure()
        dpi = f.get_dpi()
        f.set_size_inches(nx * self.magnif / dpi, ny * self.magnif / dpi)
        ax = plt.Axes(f, [0, 0, 1, 1])
        ax.set_axis_off()
        f.add_axes(ax)
        clim = (np.amin(img), np.amax(img)) if self.clim is None else self.clim

        # Plot images
        artists = []  # Will be a list of lists
        for t in range(nt):
            artist = ax.imshow(img[t], clim=clim, cmap=self.cmap,
                               animated=True)
            artists.append([artist])

        interval = 1000 / self.fps  # Milliseconds between frames
        anim = ArtistAnimation(f, artists, interval=interval)

        if savename is not None:
            anim.save(savename, writer='imagemagick')

        return anim


def imshow(img, clim=None, cmap='gray', magnif=1, savename=None):
    """Plots a borderless image.

    Args:
        img (array): NumPy array with shape (ny, nx).
        clim (tuple): Min and max image values to display.
        cmap (string): Matplotlib colormap.
        magnif (float): Factor for plotting the images at a larger/smaller
            size.
        savename (string): Optional. Name of png to save.
    """
    ny, nx = img.shape
    # Set figure
    f = plt.figure()
    dpi = f.get_dpi()
    f.set_size_inches(nx * magnif / dpi, ny * magnif / dpi)
    ax = plt.Axes(f, [0, 0, 1, 1])
    ax.set_axis_off()
    f.add_axes(ax)
    clim = (np.amin(img), np.amax(img)) if clim is None else clim

    # Plot image
    ax.imshow(img, clim=clim, cmap=cmap)

    if savename is not None:
        f.savefig(savename, dpi=dpi)


def plot_surface_points(points, savename=None, view_polar=80,
                        view_azimuthal=45, pointsize=8):
    """Plots trajectory points on the surface of the sphere.

    Args:
        points (array): A 2D array with shape (num_points, 3), containing
            points on the surface of a sphere.
        view_polar (float): Polar angle of the observer, in degrees.
        view_azimuthal (float): Azimuthal angle of the observer, in degrees.
    """
    f = plt.figure(figsize=(9, 8))
    ax = f.add_subplot(111, projection='3d')
    ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')

    lower, upper = math.floor(points.min()), math.ceil(points.max())
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_zlim(lower, upper)

    # Older versions of matplotlib might not have this
    try:
        ax.set_box_aspect((1, 1, 1))
    except:  # noqa
        pass

    ax.view_init(90 - view_polar, view_azimuthal)
    plt.draw()

    # Plot spherical surface
    radii = np.linalg.norm(points, ord=2, axis=1)
    radius = radii.mean()
    phi = np.linspace(0, 2 * pi, 100)  # Azimuthal angles
    theta = np.linspace(0, pi, 100)  # Polar angles
    x = radius * np.outer(np.cos(phi), np.sin(theta))
    y = radius * np.outer(np.sin(phi), np.sin(theta))
    z = radius * np.outer(np.ones(np.size(phi)), np.cos(theta))
    ax.plot_surface(x, y, z, color='c', alpha=0.2, shade=False)

    # Calculate points that are in front of the sphere
    points = _get_front_points(
        points, view_polar=view_polar, view_azimuthal=view_azimuthal)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               marker='.', s=pointsize, c='navy', depthshade=True)

    if savename is not None:
        f.savefig(savename, dpi=150)
        plt.close()

    return f, ax


def _get_front_points(points, view_polar=80, view_azimuthal=45):
    """Gets the points on a sphere seen by viewer of the 3D plot.

    Args:
        points (array): A 2D array with shape (num_points, 3), containing
            points on the surface of a sphere.
        view_polar (float): Polar angle of the observer, in degrees.
        view_azimuthal (float): Azimuthal angle of the observer, in degrees.

    Returns:
        points (array): A 2D array with shape (num_points, 3), containing
            points on the surface of a sphere seen by observer.
    """
    view_polar = np.radians(view_polar)
    view_azimuthal = np.radians(view_azimuthal)
    # Calculate vector pointing at observer
    v = [np.sin(view_polar) * np.cos(view_azimuthal),
         np.sin(view_polar) * np.sin(view_azimuthal),
         np.cos(view_polar)]
    v = np.array(v)

    projection = points @ v.reshape((-1, 1))
    projection = projection.reshape((-1))
    points_seen = projection > 0

    points = points[points_seen]
    return points
