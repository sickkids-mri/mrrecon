import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation


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
