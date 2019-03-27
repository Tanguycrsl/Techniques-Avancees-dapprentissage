"""A library containing generic functions."""

import matplotlib
try:
	import PyQt5
	matplotlib.use('Qt5Agg')
except ImportError:
	matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import PIL.Image

def divide_ints_check_divisible(numerator, denominator):
    """Divides the numerator by the denominator while checking that the numerator is divisible by the denominator.

    Parameters
    ----------
    numerator : int
        Numerator.
    denominator : int
        Denominator.

    Returns
    -------
    int
        Result of the division.

    Raises
    ------
    TypeError
        If `numerator` is not an instance of `int`.
    TypeError
        If `denominator` is not an instance of `int`.
    ValueError	
        If `numerator` is not divisible by `denominator`.

    """
    if not isinstance(numerator, int):
        raise TypeError('`numerator` is not an instance of `int`.')
    if not isinstance(denominator, int):
        raise TypeError('`denominator` is not an instance of `int`.')
    if numerator % denominator != 0:
        raise ValueError('`numerator` is not divisible by `denominator`.')
    return numerator//denominator

def float_to_str(float_in):
    """Converts the float into a string.

    During the conversion, "." is replaced
    by "dot" if the float is not a whole
    number and "-" is replaced by "minus".

    Parameters
    ----------
    float_in : float
        Float to be converted.

    Returns
    -------
    str
        String resulting from
        the conversion.

    """
    if float_in.is_integer():
        str_in = str(int(float_in))
    else:
        str_in = str(float_in).replace('.', 'dot')
    return str_in.replace('-', 'minus')

def histogram(data, title, path):
    """Creates a histogram of the data and saves the histogram.

    Parameters
    ----------
    data : numpy.ndarray
        1D array.
        Data.
    title : str
        Title of the histogram.
    path : str
        Path to the saved histogram. The
        path ends with ".png".

    """
    plt.hist(data, bins=60)
    plt.title(title)
    plt.savefig(path)
    plt.clf()

def plot_confusion_matrix(confusion_mat, list_names_classes, title, path):
    """Plots the confusion matrix.

    Parameters
    ----------
    confusion_mat : numpy.ndarray
        2D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        Confusion matrix to be plotted.
    list_names_classes : list
        List of class names.
    title : str
        Title of the plot.
    path : str
        Path to the saved plot.

    Raises
    ------
    TypeError
        If `confusion_mat.dtype` is not smaller than
        `numpy.integer` in the type hierarchy.

    """
    if not numpy.issubdtype(confusion_mat.dtype, numpy.integer):
        raise TypeError('`confusion_mat.dtype` is not smaller than `numpy.integer` in the type hierarchy.')
    plt.imshow(confusion_mat,
               interpolation='nearest',
               cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(list_names_classes))
    plt.xticks(tick_marks,
               list_names_classes)
    plt.yticks(tick_marks,
               list_names_classes)

    # For each type of classification (false positive,
    # false negative, ...), the number of examples in
    # this class is written.
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            plt.text(j,
                     i,
                     format(confusion_mat[i, j], 'd'),
                     horizontalalignment="center",
                     color='black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)

def plot_graphs(x_values, y_values, x_label, y_label, title, path, legend=None):
    """Overlays several graphs in the same plot and saves the plot.

    Parameters
    ----------
    x_values : numpy.ndarray
        1D array.
        x-axis values.
    y_values : numpy.ndarray
        2D array.
        `y_values[i, :]` contains the
        y-axis values of the ith graph.
    x_label : str
        x-axis label.
    y_label : str
        y-axis label.
    title : str
        Title of the plot.
    path : str
        Path to the saved plot. The path
        ends with ".png".
    legend : list, optional
        `legend[i]` is a string describing the
        ith graph. The default value is None.

    Raises
    ------
    ValueError
        If `x_values.ndim` is not equal to 1.
    ValueError
        If `y_values.ndim` is not equal to 2.

    """
    # If `x_values.ndim` and `y_values.ndim` are equal to 2
    # and `x_values.shape[0]` is equal to `y_values.shape[1]`
    # for instance, `plot_graphs` does not crash and saves
    # a wrong plot. That is why `x_values.ndim` and `y_values.ndim`
    # are checked.
    if x_values.ndim != 1:
        raise ValueError('`x_values.ndim` is not equal to 1.')
    if y_values.ndim != 2:
        raise ValueError('`y_values.ndim` is not equal to 2.')

    # Matplotlib is forced to display only
    # whole numbers on the x-axis if the
    # x-axis values are integers. Matplotlib
    # is also forced to display only whole
    # numbers on the y-axis if the y-axis
    # values are integers.
    current_axis = plt.figure().gca()
    if numpy.issubdtype(x_values.dtype, numpy.integer):
        current_axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if numpy.issubdtype(y_values.dtype, numpy.integer):
        current_axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    # For the x-axis or the y-axis, if the range
    # of the absolute values is outside [1.e-4, 1.e4],
    # scientific notation is used.
    plt.ticklabel_format(style='sci',
                         axis='both',
                         scilimits=(-4, 4))

    # `plt.plot` returns a list.
    handle = []
    for i in range(y_values.shape[0]):
        handle.append(plt.plot(x_values, y_values[i, :])[0])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend is not None:
        plt.legend(handle, legend)
    plt.savefig(path)
    plt.clf()

def save_image(path, array_uint8):
    """Saves the array as an image.

    `scipy.misc.imsave` is deprecated in Scipy 1.0.0.
    `scipy.misc.imsave` will be removed in Scipy 1.2.0.
    `save_image` replaces `scipy.misc.imsave`.

    Parameters
    ----------
    path : str
        Path to the saved image.
    array_uint8 : numpy.ndarray
        Array with data-type `numpy.uint8`.
        Array to be saved as an image.

    Raises
    ------
    TypeError
        If `array_uint8.dtype` is not equal to `numpy.uint8`.

    """
    if array_uint8.dtype != numpy.uint8:
        raise TypeError('`array_uint8.dtype` is not equal to `numpy.uint8`.')
    image = PIL.Image.fromarray(array_uint8)
    image.save(path)

def visualize_grayscale_images(images_grayscale_uint8, nb_vertically, path):
    """Arranges the grayscale images in a single image and saves the single image.

    Parameters
    ----------
    images_grayscale_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Grayscale images. `images_grayscale_uint8[i, :, :, :]`
        is the ith grayscale image. `images_grayscale_uint8.shape[3]`
        is equal to 1.
    nb_vertically : int
        Number of grayscale images per column in the single image.
    path : str
        Path to the saved single image. The path
        ends with ".png".

    Raises
    ------
    TypeError
        If `images_grayscale_uint8.dtype` is not equal
        to `numpy.uint8`.
    ValueError
        If `images_grayscale_uint8.shape[0]` is not divisible
        by `nb_vertically`.

    """
    if images_grayscale_uint8.dtype != numpy.uint8:
        raise TypeError('`images_grayscale_uint8.dtype` is not equal to `numpy.uint8`.')

    # If `images_grayscale_uint8.ndim` is not equal to 4, the
    # unpacking below raises a `ValueError` exception.
    # If `images_grayscale_uint8.shape[3]` is not equal to 1,
    # `numpy.squeeze` raises a `ValueError` exception.
    (nb_images, height_image, width_image, _) = images_grayscale_uint8.shape
    if nb_images % nb_vertically != 0:
        raise ValueError('`images_grayscale_uint8.shape[0]` is not divisible by `nb_vertically`.')

    # `nb_horizontally` has to be an integer.
    nb_horizontally = nb_images//nb_vertically
    image_uint8 = 255*numpy.ones((nb_vertically*(height_image + 1) + 1, nb_horizontally*(width_image + 1) + 1), dtype=numpy.uint8)
    for i in range(nb_vertically):
        for j in range(nb_horizontally):
            image_uint8[i*(height_image + 1) + 1:(i + 1)*(height_image + 1), j*(width_image + 1) + 1:(j + 1)*(width_image + 1)] = \
                numpy.squeeze(images_grayscale_uint8[i*nb_horizontally + j, :, :, :],
                              axis=2)
    save_image(path,
               image_uint8)


