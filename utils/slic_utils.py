
from skimage.segmentation.slic_superpixels import _get_mask_centroids
from skimage.segmentation._slic import (_slic_cython, _enforce_label_connectivity_cython)
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, regular_grid
from skimage.color import rgb2lab
from skimage.measure import label, regionprops
from skimage import io

import os.path
import warnings
import functools
import collections as coll
import numpy as np
from numpy import random
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.vq import kmeans2


def slic_custom(image, n_segments=100, compactness=10., max_iter=10, sigma=0,
                spacing=None, multichannel=True, convert2lab=None,
                enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3,
                slic_zero=False, start_label=None, mask=None, previous_centroids=None):
    """Segments image using k-means clustering in Color-(x,y,z) space.

    Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or multichannel
        (see `multichannel` parameter).
    n_segments : int, optional
        The (approximate) number of labels in the segmented output image.
    compactness : float, optional
        Balances color proximity and space proximity. Higher values give
        more weight to space proximity, making superpixel shapes more
        square/cubic. In SLICO mode, this is the initial compactness.
        This parameter depends strongly on image contrast and on the
        shapes of objects in the image. We recommend exploring possible
        values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before
        refining around a chosen value.
    max_iter : int, optional
        Maximum number of iterations of k-means.
    sigma : float or (3,) array-like of floats, optional
        Width of Gaussian smoothing kernel for pre-processing for each
        dimension of the image. The same sigma is applied to each dimension in
        case of a scalar value. Zero means no smoothing.
        Note, that `sigma` is automatically scaled if it is scalar and a
        manual voxel spacing is provided (see Notes section).
    spacing : (3,) array-like of floats, optional
        The voxel spacing along each image dimension. By default, `slic`
        assumes uniform spacing (same voxel resolution along z, y and x).
        This parameter controls the weights of the distances along z, y,
        and x during k-means clustering.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    convert2lab : bool, optional
        Whether the input should be converted to Lab colorspace prior to
        segmentation. The input image *must* be RGB. Highly recommended.
        This option defaults to ``True`` when ``multichannel=True`` *and*
        ``image.shape[-1] == 3``.
    enforce_connectivity : bool, optional
        Whether the generated segments are connected or not
    min_size_factor : float, optional
        Proportion of the minimum segment size to be removed with respect
        to the supposed segment size ```depth*width*height/n_segments```
    max_size_factor : float, optional
        Proportion of the maximum connected segment size. A value of 3 works
        in most of the cases.
    slic_zero : bool, optional
        Run SLIC-zero, the zero-parameter mode of SLIC. [2]_
    start_label: int, optional
        The labels' index start. Should be 0 or 1.
    mask : 2D ndarray, optional
        If provided, superpixels are computed only where mask is True,
        and seed points are homogeneously distributed over the mask
        using a K-means clustering strategy.

    Returns
    -------
    labels : 2D or 3D array
        Integer mask indicating segment labels.

    Raises
    ------
    ValueError
        If ``convert2lab`` is set to ``True`` but the last array
        dimension is not of length 3.
    ValueError
        If ``start_label`` is not 0 or 1.

    Notes
    -----
    * If `sigma > 0`, the image is smoothed using a Gaussian kernel prior to
      segmentation.

    * If `sigma` is scalar and `spacing` is provided, the kernel width is
      divided along each dimension by the spacing. For example, if ``sigma=1``
      and ``spacing=[5, 1, 1]``, the effective `sigma` is ``[0.2, 1, 1]``. This
      ensures sensible smoothing for anisotropic images.

    * The image is rescaled to be in [0, 1] prior to processing.

    * Images of shape (M, N, 3) are interpreted as 2D RGB images by default. To
      interpret them as 3D with the last dimension having length 3, use
      `multichannel=False`.

    * `start_label` is introduced to handle the issue [4]_. The labels
      indexing starting at 0 will be deprecated in future versions. If
      `mask` is not `None` labels indexing starts at 1 and masked area
      is set to 0.

    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.
        :DOI:`10.1109/TPAMI.2012.120`
    .. [2] https://www.epfl.ch/labs/ivrl/research/slic-superpixels/#SLICO
    .. [3] Irving, Benjamin. "maskSLIC: regional superpixel generation with
           application to local pathology characterisation in medical images.",
           2016, :arXiv:`1606.09518`
    .. [4] https://github.com/scikit-image/scikit-image/issues/3722

    Examples
    --------
    >>> from skimage.segmentation import slic
    >>> from skimage.data import astronaut
    >>> img = astronaut()
    >>> segments = slic(img, n_segments=100, compactness=10)

    Increasing the compactness parameter yields more square regions:

    >>> segments = slic(img, n_segments=100, compactness=20)

    """

    image = img_as_float(image)
    use_mask = mask is not None
    dtype = image.dtype

    is_2d = False

    if image.ndim == 2:
        # 2D grayscale image
        image = image[np.newaxis, ..., np.newaxis]
        is_2d = True
    elif image.ndim == 3 and multichannel:
        # Make 2D multichannel image 3D with depth = 1
        image = image[np.newaxis, ...]
        is_2d = True
    elif image.ndim == 3 and not multichannel:
        # Add channel as single last dimension
        image = image[..., np.newaxis]

    if multichannel and (convert2lab or convert2lab is None):
        if image.shape[-1] != 3 and convert2lab:
            raise ValueError("Lab colorspace conversion requires a RGB image.")
        elif image.shape[-1] == 3:
            image = rgb2lab(image)

    if start_label is None:
        if use_mask:
            start_label = 1
        else:
            warnings.warn("skimage.measure.label's indexing starts from 0. " +
                          "In future version it will start from 1. " +
                          "To disable this warning, explicitely " +
                          "set the `start_label` parameter to 1.",
                          FutureWarning, stacklevel=2)
            start_label = 0

    if start_label not in [0, 1]:
        raise ValueError("start_label should be 0 or 1.")

    # initialize cluster centroids for desired number of segments
    update_centroids = False
    if use_mask:
        mask = np.ascontiguousarray(mask, dtype=np.bool).view('uint8')
        if mask.ndim == 2:
            mask = np.ascontiguousarray(mask[np.newaxis, ...])
        if mask.shape != image.shape[:3]:
            raise ValueError("image and mask should have the same shape.")
        #centroids, steps = _get_mask_centroids(mask, n_segments)
        centroids, steps = _get_mask_centroids(image, n_segments, multichannel=multichannel) # for hs3090
        update_centroids = True
    else:
        centroids, steps = _get_grid_centroids(image, n_segments)
        #centroids, steps = _get_grid_centroids(image, n_segments, multichannel=multichannel)

    # set previous centroids to initial cetroids
    if previous_centroids is not None:
        centroids = np.zeros((previous_centroids.shape[1],3))
        centroids[:,1:] = previous_centroids[0][:]
        # import pdb; pdb.set_trace()

    if spacing is None:
        spacing = np.ones(3, dtype=dtype)
    elif isinstance(spacing, (list, tuple)):
        spacing = np.ascontiguousarray(spacing, dtype=dtype)

    if not isinstance(sigma, coll.Iterable):
        sigma = np.array([sigma, sigma, sigma], dtype=dtype)
        sigma /= spacing.astype(dtype)
    elif isinstance(sigma, (list, tuple)):
        sigma = np.array(sigma, dtype=dtype)
    if (sigma > 0).any():
        # add zero smoothing for multichannel dimension
        sigma = list(sigma) + [0]
        image = ndi.gaussian_filter(image, sigma)

    n_centroids = centroids.shape[0]

    segments = np.ascontiguousarray(np.concatenate(
        [centroids, np.zeros((n_centroids, image.shape[3]))],
        axis=-1), dtype=dtype)

    # Scaling of ratio in the same way as in the SLIC paper so the
    # values have the same meaning
    step = max(steps)
    ratio = 1.0 / compactness

    image = np.ascontiguousarray(image * ratio, dtype=dtype)

    if update_centroids:
        # Step 2 of the algorithm [3]_
        _slic_cython(image, mask, segments, step, max_iter, spacing,
                     slic_zero, ignore_color=True,
                     start_label=start_label)

    labels = _slic_cython(image, mask, segments, step, max_iter,
                          spacing, slic_zero, ignore_color=False,
                          start_label=start_label)

    if enforce_connectivity:
        if use_mask:
            segment_size = mask.sum() / n_centroids
        else:
            segment_size = np.prod(image.shape[:3]) / n_centroids
        min_size = int(min_size_factor * segment_size)
        max_size = int(max_size_factor * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels, min_size, max_size, start_label=start_label)

    if is_2d:
        labels = labels[0]

    # obtain current centroids
    props = regionprops(labels) # 3
    points = np.zeros((len(props),2))
    # label 순으로 centroids 정렬
    for idx, prop in enumerate(props):
        points[prop.label-1] = prop.centroid
    centroids = np.array(points)[np.newaxis,]  # 1 3 2
    closest_indices = None
    if previous_centroids is not None:
        points_prev = previous_centroids[0].copy()  # 6, 2
        points_repeat = np.repeat(np.array(points)[:, np.newaxis], previous_centroids.shape[0], axis=1)  # 3, 1, 2 -> 3, 6, 2
        closest_indices = np.argmin(np.linalg.norm(points_repeat - points_prev[np.newaxis,], axis=2), axis=1) # 1 6 2

    return labels, centroids, closest_indices

