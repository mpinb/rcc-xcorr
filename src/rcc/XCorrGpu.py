import math

import GPUtil

import numpy as np
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

    def __init__(self, crop_output=(0, 0), normalize_input=False, cache_correlation=False):
        self.correlation = None
        self.crop_output = crop_output
        self.normalize_input = normalize_input
        self.cache_correlation = cache_correlation
        # NOTE: gpu util uses nvidia-smi to set the cuda device count
        self.cuda_devices = GPUtil.getAvailable(order='first', limit=4)
        self.num_devices = len(self.cuda_devices)
        print(f'Using {self.num_devices} CUDA devices: {self.cuda_devices} ')

    # XCorrGpu info
    def description(self):
        return f"[XCorrGpu] normalize_input:{self.normalize_input}"

    # Return the previously computed correlation
    # if the cache_correlation flag is True
    def get_correlation(self):
        return self.correlation if self.cache_correlation else None

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
    def norm_xcorr(self, image, template, mode='constant', constant_values=0):
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

    def norm_xcorr_array(self, image, template_array, mode='constant', constant_values=0):

        float_dtype = image.dtype
        image_shape = image.shape
        template_array_size = len(template_array)
        template_shape = template_array[0].shape

        if image.ndim != 2 or any(template_array[x].ndim != 2 for x in range(template_array_size)):
            raise ValueError("Dimensionality of image and/or templates should be 2.")

        if np.any(np.less(image_shape, template_shape)):
            raise ValueError("Image size must be larger than template size.")

        if any(template_array[x].shape != template_shape for x in range(template_array_size)):
            raise ValueError("All templates in correlation group should have the same size.")

        # Padding image
        pad_width = tuple((width, width) for width in template_shape)
        if mode == 'constant':
            image = cp.pad(image, pad_width=pad_width, mode=mode,
                           constant_values=constant_values)
        else:
            image = cp.pad(image, pad_width=pad_width, mode=mode)

        # Compute image_window sums (used to normalize the output)
        image_window_sum = _window_sum(image, template_shape)
        image_window_sum2 = _window_sum(image ** 2, template_shape)

        template_mean_list = [template.mean() for template in template_array]
        template_area = math.prod(template_shape)
        template_ssd_list = [cp.sum((template - template_mean) ** 2)
                             for template, template_mean
                             in zip(template_array, template_mean_list)]

        # Flipping template to make convolution equivalent to cross correlation
        xcorr_list = [fftconvolve(image, template[::-1, ::-1],
                            mode="valid")[1:-1, 1:-1]
                      for template in template_array]

        # Computing the numerator
        numerator_list = [xcorr - image_window_sum * template_mean
                          for xcorr, template_mean
                          in zip(xcorr_list, template_mean_list)]

        # Computing the denominator common part
        denominator_common = image_window_sum2
        cp.multiply(image_window_sum, image_window_sum, out=image_window_sum)
        cp.divide(image_window_sum, template_area, out=image_window_sum)
        denominator_common -= image_window_sum

        # Computing the denominator individual part
        denominator_list = [denominator_common * template_ssd for template_ssd in template_ssd_list]
        # NOTE: Using 0 for the maximum to avoid taking the sqrt of a negative number
        [cp.maximum(denominator, 0, out=denominator) for denominator in denominator_list]
        [cp.sqrt(denominator, out=denominator) for denominator in denominator_list]

        # Computing the output (normalized fast cross correlation)
        norm_xcorr_list = [cp.zeros_like(xcorr, dtype=float_dtype) for xcorr in xcorr_list]

        # avoid zero-division
        mask_list = [denominator > cp.finfo(float_dtype).eps for denominator in denominator_list]

        for indx in range(template_array_size):
            mask = mask_list[indx]
            numerator = numerator_list[indx]
            denominator = denominator_list[indx]
            (norm_xcorr_list[indx])[mask] = numerator[mask] / denominator[mask]

        return norm_xcorr_list

    # fast normalized cross-correlation
    def match_template(self, image, template, correlation_num):
        # using correlation_number to assign cuda device (round robin)
        cuda_device = self.cuda_devices[correlation_num % self.num_devices]
        with cp.cuda.Device(cuda_device):
            image_gpu = cp.asarray(image)
            template_gpu = cp.asarray(template)

            if self.normalize_input:
                image_gpu -= image_gpu.mean()
                template_gpu -= template_gpu.mean()

            norm_xcorr = self.norm_xcorr(image_gpu, template_gpu, mode='constant', constant_values=0)

            # cropping the norm_xcorr
            cropy, cropx = self.crop_output
            origy, origx = norm_xcorr.shape
            norm_xcorr = norm_xcorr[cropy:origy-cropy,cropx:origx-cropx]

            # cache correlation
            if self.cache_correlation:
                self.correlation = norm_xcorr

            # NOTE: argmax returns the first occurrence of the maximum value
            xcorr_peak = cp.argmax(norm_xcorr)
            y, x = cp.unravel_index(xcorr_peak, norm_xcorr.shape)  # (correlation peak coordinates)

            return y.get() + cropy, x.get() + cropx, norm_xcorr[y,x].get()

    # fast normalized cross-correlation
    def match_template_array(self, image, template_list, corr_list, corr_list_num):
        # using correlation_list_number to assign cuda device (round robin)
        cuda_device = self.cuda_devices[corr_list_num % self.num_devices]
        with cp.cuda.Device(cuda_device):
            num_templates = len(template_list)
            image_gpu = cp.asarray(image)

            if self.normalize_input:
                image_gpu -= image_gpu.mean()

            templates_array = []
            for indx in range(num_templates):
                template_gpu = cp.asarray(template_list[indx])

                if self.normalize_input:
                    template_gpu -= template_gpu.mean()

                templates_array.append(template_gpu)

            norm_xcorr_list = self.norm_xcorr_array(image_gpu, templates_array, mode='constant', constant_values=0)

            # cropping the correlations
            cropy, cropx = self.crop_output
            origy, origx = norm_xcorr_list[0].shape
            norm_xcorr_list = [norm_xcorr[cropy:origy-cropy, cropx:origx-cropx] for norm_xcorr in norm_xcorr_list]

            # cache correlation
            if self.cache_correlation:
                pass  # ignoring this flag for grouped correlations

            match_results_coord = np.empty((0, 3), int)
            match_results_peak = np.empty((0, 2), float)

            for indx in range(num_templates):
                norm_xcorr = norm_xcorr_list[indx]
                # NOTE: argmax returns the first occurrence of the maximum value
                xcorr_peak = cp.argmax(norm_xcorr)
                y, x = cp.unravel_index(xcorr_peak, norm_xcorr.shape)  # (correlation peak coordinates)
                match_result_coord = np.array([[corr_list[indx], y.get(), x.get()]])
                match_result_peak = np.array([[corr_list[indx], norm_xcorr[y,x].get()]])
                match_results_coord = np.append(match_results_coord, match_result_coord, axis=0)
                match_results_peak = np.append(match_results_peak, match_result_peak, axis=0)

            return match_results_coord, match_results_peak
