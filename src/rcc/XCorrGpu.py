import math

import cupy as cp
from cupyx.scipy.signal import fftconvolve


# We thank Eli Horn for providing this code, used with his permission,
# to speed up the calculation of local sums. The algorithm depends on
# precomputing running sums as described in "Fast Normalized
# Cross-Correlation", by J. P. Lewis, Industrial Light & Magic.
# http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html
def local_sum(A, m, n):
    B = cp.lib.pad(A, ((m, m), (n, n)), 'constant', constant_values=0)
    s = cp.cumsum(B, axis=0)
    c = s[m:-1, :] - s[:-m - 1, :]
    s = cp.cumsum(c, axis=1)
    return s[:, n:-1] - s[:, :-n - 1]


# Perform fast cross correlation using method from: J.P. Lewis (Industrial Light & Magic)
# REF: http://scribblethink.org/Work/nvisionInterface/nip.pdf
def _window_sum(image, window_shape):
    window_sum = cp.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1]
                  - window_sum[:-window_shape[0] - 1])

    window_sum = cp.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1]
                  - window_sum[:, :-window_shape[1] - 1])

    return window_sum


# no preprocessing (mean subtraction , std dev.) (raw)
# full alignment for the 3D alignment only edges overlap (mode = 'full')
class XCorrGpu:

    def __init__(self, normalize_output=True, normalize_input=False):
        self.normalize_output = normalize_output
        self.normalize_input = normalize_input

    # XCorrGpu info
    def description(self):
        return f"[XCorrGpu] normalize_output:{self.normalize_output}"

    """cross correlate template to a 2-D image using fast normalized correlation.
    The output is an array with values between -1.0 and 1.0. The value at a
    given position corresponds to the correlation coefficient between the image
    and the template.
    Parameters
    ----------
    image : (M, N) array
        2-D input image.
    template : (m, n) array
        Template to locate. It must be `(m <= M, n <= N)`.
    pad_input : bool
        If True, pad `image` so that output is the same size as the image, and
        output values correspond to the template center.
    mode : see `numpy.pad`, optional
        Padding mode.
    constant_values : see `numpy.pad`, optional
        Constant values used in conjunction with ``mode='constant'``.
    Returns
    -------
    output : array
        Response image with correlation coefficients.
    Notes
    -----
    Details on the cross-correlation are presented in [1]_. This implementation
    uses FFT convolutions of the image and the template.
    References
    ----------
    .. [1] J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light
           and Magic.
    """
    def fast_xcorr(self, image, template, mode='constant', constant_values=0):
        if image.ndim != 2 or template.ndim != 2:
            raise ValueError("Dimensionality of image and/or template should be 2.")
        #if cp.any(cp.less(image.shape, template.shape)):
        #    raise ValueError("Image must be larger than template.")

        float_dtype = image.dtype
        image_shape = image.shape

        pad_width = tuple((width, width) for width in template.shape)
        if mode == 'constant':
            image = cp.pad(image, pad_width=pad_width, mode=mode,
                           constant_values=constant_values)
        else:
            image = cp.pad(image, pad_width=pad_width, mode=mode)

        # Compute image_window sums (used to normalize the output)
        image_window_sum = _window_sum(image, template.shape)
        image_window_sum2 = _window_sum(image ** 2, template.shape)

        template_mean = template.mean()
        template_area = math.prod(template.shape)
        template_ssd = cp.sum((template - template_mean) ** 2)

        # Flipping template to make convolution equivalent to cross correlation
        xcorr = fftconvolve(image, template[::-1, ::-1],
                            mode="valid")[1:-1, 1:-1]

        numerator = xcorr - image_window_sum * template_mean

        denominator = image_window_sum2
        cp.multiply(image_window_sum, image_window_sum, out=image_window_sum)
        cp.divide(image_window_sum, template_area, out=image_window_sum)
        denominator -= image_window_sum
        denominator *= template_ssd
        cp.maximum(denominator, 0, out=denominator)  # sqrt of negative number not allowed
        cp.sqrt(denominator, out=denominator)

        response = cp.zeros_like(xcorr, dtype=float_dtype)

        # avoid zero-division
        mask = denominator > cp.finfo(float_dtype).eps

        response[mask] = numerator[mask] / denominator[mask]

        return response

    # fast cross-correlation
    # no preprocessing (mean subtraction , std dev.) (raw)
    def match_template(self, image, template):

        image_gpu = cp.asarray(image)
        template_gpu = cp.asarray(template)

        if self.normalize_input:
            image_gpu -= image_gpu.mean()
            template_gpu -= template_gpu.mean()

        corr = self.fast_xcorr(image_gpu, template_gpu, mode='constant', constant_values=0)


        #NOTE: argmax returns the first occurence of the maximum value
        corr_peak = cp.argmax(corr)
        y, x = cp.unravel_index(corr_peak, corr.shape)  # (correlation peak coordinates)

        #print(f"Correlation type: {corr.dtype}  with {np.argmax(corr)} and {corr[y,x]}")

        return y.get(), x.get(), corr[y,x].get()
