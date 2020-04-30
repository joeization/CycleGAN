import numpy as np
from numpy.core.umath_tests import inner1d
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def image_histogram_equalization(image, number_bins=256):
    '''histogram equalization the image
    '''
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(
        image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)  # , cdf


def elastic_transform(image, alpha=512, sigma=20, spline_order=1, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    image = image.reshape((256, 512, 1))
    assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    return result


def center_crop(layer, target_size, target_size2):
    _, _, layer_width, layer_height = layer.size()
    xy1 = (layer_width - target_size) // 2
    xy2 = (layer_height - target_size2) // 2
    return layer[:, :, xy1:(xy1 + target_size), xy2:(xy2 + target_size2)]


def pixel_list(im):
    ret = []
    i = 0
    for x in im:
        j = 0
        for y in x:
            if y > 0:
                ret.append([i, j])
            j += 1
        i += 1
    return np.array(ret)


def HausdorffDist(A, B):
    # Hausdorf Distance: Compute the Hausdorff distance between two point
    # clouds.
    # Let A and B be subsets of metric space (Z,dZ),
    # The Hausdorff distance between A and B, denoted by dH(A,B),
    # is defined by:
    # dH(A,B) = max(h(A,B),h(B,A)),
    # where h(A,B) = max(min(d(a,b))
    # and d(a,b) is a L2 norm
    # dist_H = hausdorff(A,B)
    # A: First point sets (MxN, with M observations in N dimension)
    # B: Second point sets (MxN, with M observations in N dimension)
    # ** A and B may have different number of rows, but must have the same
    # number of columns.
    #
    # Edward DongBo Cui; Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T +
                    inner1d(B, B)-2*(np.dot(A, B.T)))
    # Find DH
    dH = np.max(
        np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))
    return(dH)


def get_n_fold(total, fold, idx):
    if len(total) % fold != 0 or idx < 0 or idx >= fold:
        raise ValueError
    fd = total[idx::fold]
    for f in fd:
        total.remove(f)
    return fd


if __name__ == "__main__":
    from PIL import Image
    from matplotlib import pyplot as plt
    prev_mask = Image.open('./data/ultrasound/ground truth/G0/01/0000.png')
    prev_mask = elastic_transform(
        np.array(prev_mask)).reshape(256, 512)
    prev_mask = Image.fromarray(prev_mask)
    plt.imshow(prev_mask, cmap='gray')
    plt.show()
