import numpy as np

from skimage.feature import _hoghistogram
# from skimage import _hoghistogram
# from hog_utils import hog_utils as utils
from sides.hog_utils import hog_utils as utils


def _hog_normalize_block(block, method, eps=1e-5):
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L1-sqrt':
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block**2) + eps**2)
    elif method == 'L2-Hys':
        out = block / np.sqrt(np.sum(block**2) + eps**2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out**2) + eps**2)
    else:
        raise ValueError('Selected block normalization method is invalid.')

    return out


def _hog_channel_gradient(channel):
    """Compute unnormalized gradient image along `row` and `col` axes.

    Parameters
    ----------
    channel : (M, N) ndarray
        Grayscale image or one of image channel.

    Returns
    -------
    g_row, g_col : channel gradient along `row` and `col` axes correspondingly.
    """
    g_row = np.empty(channel.shape, dtype=channel.dtype)
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
    g_col = np.empty(channel.shape, dtype=channel.dtype)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]

    return g_row, g_col


# @utils.channel_as_last_axis(multichannel_output=False)
def hog(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),
    block_norm='L2-Hys',
    visualize=False,
    transform_sqrt=False,
    feature_vector=True,
    *,
    channel_axis=None,
):
    image = np.atleast_2d(image)
    float_dtype = utils._supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    multichannel = channel_axis is not None
    ndim_spatial = image.ndim - 1 if multichannel else image.ndim
    if ndim_spatial != 2:
        raise ValueError(
            'Only images with two spatial dimensions are '
            'supported. If using with color/multichannel '
            'images, specify `channel_axis`.'
        )

    """
    The first stage applies an optional global image normalization
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each color channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

    if transform_sqrt:
        image = np.sqrt(image)

    """
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    color channel is used, which provides color invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    if multichannel:
        g_row_by_ch = np.empty_like(image, dtype=float_dtype)
        g_col_by_ch = np.empty_like(image, dtype=float_dtype)
        g_magn = np.empty_like(image, dtype=float_dtype)

        for idx_ch in range(image.shape[2]):
            (
                g_row_by_ch[:, :, idx_ch],
                g_col_by_ch[:, :, idx_ch],
            ) = _hog_channel_gradient(image[:, :, idx_ch])
            g_magn[:, :, idx_ch] = np.hypot(
                g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch]
            )

        # For each pixel select the channel with the highest gradient magnitude
        idcs_max = g_magn.argmax(axis=2)
        rr, cc = np.meshgrid(
            np.arange(image.shape[0]),
            np.arange(image.shape[1]),
            indexing='ij',
            sparse=True,
        )
        g_row = g_row_by_ch[rr, cc, idcs_max]
        g_col = g_col_by_ch[rr, cc, idcs_max]
    else:
        g_row, g_col = _hog_channel_gradient(image)

    """
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    s_row, s_col = image.shape[:2]
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis

    # compute orientations integral images
    orientation_histogram = np.zeros(
        (n_cells_row, n_cells_col, orientations), dtype=float
    )
    g_row = g_row.astype(float, copy=False)
    g_col = g_col.astype(float, copy=False)

    _hoghistogram.hog_histograms(
        g_col,
        g_row,
        c_col,
        c_row,
        s_col,
        s_row,
        n_cells_col,
        n_cells_row,
        orientations,
        orientation_histogram,
    )

  
    return orientation_histogram
