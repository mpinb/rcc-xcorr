import math
import os
import sys
import time
import logging
import contextlib

import GPUtil

import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
handler = logging.StreamHandler()
handler.setStream(sys.stdout)
logger.addHandler(handler)

# Attempting to load nvtx (used for profiler annotations)
try:
    import nvtx
except ImportError as ie:
    logger.warn(f"Unable to import nvtx library. {ie}")
    nvtx_available = False
else:
    nvtx_available = True

# Conditional annotate code during performance profiling
# based on the presence of the nvtx annotation library
def conditional_annotate(nvtx_available, label, color):
    def decorator(func):
        ctx = nvtx.annotate(label, color=color) \
            if nvtx_available else contextlib.suppress()
        with ctx:
            result = func
        return result
    return decorator


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

    def __init__(self,
                 normalize_input=False,
                 crop_output=(0, 0),
                 override_eps=False,
                 custom_eps=1e-6,
                 cache_correlation=False,
                 max_devices=1):
        self.correlation = None
        self.crop_output = crop_output
        self.override_eps = override_eps
        self.custom_eps = custom_eps
        self.normalize_input = normalize_input
        self.cache_correlation = cache_correlation
        attempts = 5  # NOTE: attempts to use the GPU
        for i in range(attempts):
            cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
            if cuda_visible_devices:  # Using CUDA_VISIBLE_DEVICES to set the cuda devices (if present)
                logger.debug(f'Using CUDA_VISIBLE_DEVICES: {cuda_visible_devices}')
                visible_devices = [int(d) for d in cuda_visible_devices.split(",")]
                self.cuda_devices = [d for d in range(cp.cuda.runtime.getDeviceCount())]
            else:   # GPUtil uses nvidia-smi to set the available cuda devices
                self.cuda_devices = GPUtil.getAvailable(order='first', limit=max_devices, maxLoad=0.5, maxMemory=0.5)
                visible_devices = self.cuda_devices
            self.num_devices = len(self.cuda_devices)
            if self.num_devices:
                logger.debug(f'[PID: {os.getpid()}] Using {self.num_devices} CUDA device(s): {visible_devices} ')
                gpus = GPUtil.getGPUs()
                gpus = [gpus[g] for g in visible_devices]
                for gpu in gpus:
                    logger.info(f'[PID: {os.getpid()}] GPU: {gpu.id}, Available memory: {gpu.memoryFree} MB')
                break
            # If not the last attempt, sleep for 5 seconds and retry
            if i != attempts-1:
                GPUtil.showUtilization()
                time.sleep(10)
            else:
                raise RuntimeError(f'Could not find an available GPU after {attempts} attempts.')
        # Setting the fft plan cache to zero (avoid allocating GPU memory)
        # cp.fft.config.set_plan_cache_size(0)

    # XCorrGpu info
    def description(self):
        return f"XCorrGpu(normalize_input:{self.normalize_input}, crop_output:{self.crop_output})"

    def cleanup(self, worker_id=0):
        # show usage of fft plans
        #logger.info(f'XCorrGpu::cleanup(TID:{current_thread().name})')
        #cp.fft.config.show_plan_cache_info()

        # cleanup device allocated resources (per thread)
        for cuda_device in self.cuda_devices:
            with cp.cuda.Device(cuda_device):
                # cleanup fft cache memory
                cache = cp.fft.config.get_plan_cache()
                #print(f'XCorrGpu(TID:{current_thread().name})', "before clearing cache:", cp.get_default_memory_pool().used_bytes()/1024, "kB")
                cache.clear()
                #print(f'XCorrGpu(TID:{current_thread().name})', "after clearing cache:", cp.get_default_memory_pool().used_bytes()/1024, "kB")

                # free allocated memory
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

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
    ##@nvtx.annotate("norm_xcorr()", color="green")
    @conditional_annotate(nvtx_available, label="norm_xcorr()", color="green")
    def norm_xcorr(self, image, template, mode='constant', constant_values=0):
        if image.ndim != 2 or template.ndim != 2:
            raise ValueError("Dimensionality of image and/or template should be 2.")
        #if cp.any(cp.less(image.shape, template.shape)):
        #    raise ValueError("Image must be larger than template.")

        float_dtype = image.dtype
        image_shape = image.shape
        small_value = self.custom_eps if self.override_eps else cp.finfo(float_dtype).eps

        # Padding image
        # with nvtx.annotate("padding image", color="blue"):
        pi_ctx = nvtx.annotate("padding image", color="blue") \
            if nvtx_available else contextlib.suppress()
        with pi_ctx:
            pad_width = tuple((width, width) for width in template.shape)
            if mode == 'constant':
                image = cp.pad(image, pad_width=pad_width, mode=mode,
                            constant_values=constant_values)
            else:
                image = cp.pad(image, pad_width=pad_width, mode=mode)

        # Compute image_window sums (used to normalize the output)
        # with nvtx.annotate("compute window sums", color="yellow"):
        cws_ctx = nvtx.annotate("compute window sums", color="yellow") \
            if nvtx_available else contextlib.suppress()
        with cws_ctx:
            image_window_sum = _window_sum(image, template.shape)
            image_window_sum2 = _window_sum(image ** 2, template.shape)

        # Compute template square sum differences
        # with nvtx.annotate("template sq. sum diff.", color="blue"):
        tssd_ctx = nvtx.annotate("template sq. sum diff.", color="blue") \
            if nvtx_available else contextlib.suppress()
        with tssd_ctx:
            template_mean = template.mean()
            template_area = math.prod(template.shape)
            template_ssd = cp.sum((template - template_mean) ** 2)

        # Flipping template to make convolution equivalent to cross correlation
        # with nvtx.annotate("fft convolve", color="orange"):
        fft_ctx = nvtx.annotate("fft convolve", color="orange") \
            if nvtx_available else contextlib.suppress()
        with fft_ctx:
            xcorr = fftconvolve(image, template[::-1, ::-1],
                                mode="valid")[1:-1, 1:-1]

        # Compute numerator
        # with nvtx.annotate("compute numerator", color="pink"):
        cn_ctx = nvtx.annotate("compute numerator", color="pink") \
            if nvtx_available else contextlib.suppress()
        with cn_ctx:
            numerator = xcorr - image_window_sum * template_mean

        # Compute denominator
        # with nvtx.annotate("compute denominator", color="brown"):
        cd_ctx = nvtx.annotate("compute denominator", color="brown") \
            if nvtx_available else contextlib.suppress()
        with cd_ctx:
            denominator = image_window_sum2
            cp.multiply(image_window_sum, image_window_sum, out=image_window_sum)
            cp.divide(image_window_sum, template_area, out=image_window_sum)
            denominator -= image_window_sum
            denominator *= template_ssd
            cp.maximum(denominator, 0, out=denominator)  # sqrt of negative number not allowed
            cp.sqrt(denominator, out=denominator)

        # Compute response
        # with nvtx.annotate("compute response", color="yellow"):
        cr_ctx = nvtx.annotate("compute response", color="yellow") \
            if nvtx_available else contextlib.suppress()
        with cr_ctx:
            response = cp.zeros_like(xcorr, dtype=float_dtype)
            cp.divide(numerator, denominator, out=response)
            cp.putmask(response, small_value > denominator, 0)
            #cp.putmask(response, cp.isinf(response), 0)
            #cp.putmask(response, cp.isnan(response), 0)

        return response

    ##@nvtx.annotate("norm_xcorr_array()", color="green")
    @conditional_annotate(nvtx_available, label="norm_xcorr_array()", color="green")
    def norm_xcorr_array(self, image, template_array, mode='constant', constant_values=0):

        float_dtype = image.dtype
        image_shape = image.shape
        template_array_size = len(template_array)
        template_shape = template_array[0].shape
        small_value = self.custom_eps if self.override_eps else cp.finfo(float_dtype).eps

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

        # Computing the response
        for indx in range(template_array_size):
            numerator = numerator_list[indx]
            denominator = denominator_list[indx]
            cp.divide(numerator, denominator, out=norm_xcorr_list[indx])
            cp.putmask(norm_xcorr_list[indx], small_value > denominator, 0)

        return norm_xcorr_list

    # fast normalized cross-correlation
    #@nvtx.annotate('match_template()', color="magenta")
    @conditional_annotate(nvtx_available, label="match_template()", color="magenta")
    def match_template(self, image, template, correlation_num=1):

        # using correlation_number to assign cuda device (round robin)
        cuda_device = self.cuda_devices[correlation_num % self.num_devices]

        logger.debug(f'[PID: {os.getpid()}] image: {image.shape}, template: {template.shape}')
        logger.debug(f'[PID: {os.getpid()}] cuda_device: {cuda_device}, corr_num: {correlation_num}')

        with cp.cuda.Device(cuda_device):
            image_gpu = cp.asarray(image)
            template_gpu = cp.asarray(template)

            if self.normalize_input:
                image_gpu -= image_gpu.mean()
                template_gpu -= template_gpu.mean()

            # the normalization occurs in the GPU
            norm_xcorr = self.norm_xcorr(image_gpu, template_gpu, mode='constant', constant_values=0)

            # show usage of fft plans
            #logger.info(f'[XCorrGpu(PID: {os.getpid()}, TID: {current_thread().name})] show plan cache info.')
            #cp.fft.config.show_plan_cache_info()

            # cropping the norm_xcorr
            cropx, cropy = self.crop_output
            origx, origy = norm_xcorr.shape
            norm_xcorr = norm_xcorr[cropx:origx - cropx, cropy:origy - cropy]

            # cache cropped correlation
            if self.cache_correlation:
                self.correlation = norm_xcorr

            # finding the peak value inside the norm xcorr (cupy)
            # with nvtx.annotate("find max/coords cupy", color="blue"):
            fmc_ctx = nvtx.annotate("find max/coords cupy", color="blue") \
                if nvtx_available else contextlib.suppress()
            with fmc_ctx:
                # NOTE: argmax returns the first occurrence of the maximum value
                #xcorr_peak = cp.argmax(np.where(np.isfinite(norm_xcorr), norm_xcorr, 0))
                xcorr_peak = cp.argmax(norm_xcorr)
                y, x = cp.unravel_index(xcorr_peak, norm_xcorr.shape)  # (correlation peak coordinates)

            return y.get() + cropy, x.get() + cropx, norm_xcorr[y,x].get()

            # # finding the peak value inside the norm xcorr (numpy)
            # # NOTE: numpy.isfinite test for both finite and NaN
            # with nvtx.annotate("find max/coords numpy", color="green"):
            #     xcorr_peak = np.argmax(np.where(np.isfinite(norm_xcorr), norm_xcorr, 0))
            #     y, x = np.unravel_index(xcorr_peak, norm_xcorr.shape)

            # return y + cropy, x + cropx, norm_xcorr[y,x]


    # fast normalized cross-correlation
    ##@nvtx.annotate('match_template_array()', color="magenta")
    @conditional_annotate(nvtx_available, label="match_template_array()", color="magenta")
    def match_template_array(self, image, template_list, corr_list, corr_list_num=1):
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
            cropx, cropy = self.crop_output
            origx, origy = norm_xcorr_list[0].shape
            norm_xcorr_list = [norm_xcorr[cropx:origx - cropx, cropy:origy - cropy] for norm_xcorr in norm_xcorr_list]

            # cache correlation
            if self.cache_correlation:
                pass  # ignoring this flag for grouped correlations

            match_results_coord = np.empty((0, 3), int)
            match_results_peak = np.empty((0, 2), float)

            for indx in range(num_templates):
                norm_xcorr = norm_xcorr_list[indx]
                # NOTE: argmax returns the first occurrence of the maximum value
                #xcorr_peak = cp.argmax(np.where(np.isfinite(norm_xcorr), norm_xcorr, 0))
                xcorr_peak = cp.argmax(norm_xcorr)
                y, x = cp.unravel_index(xcorr_peak, norm_xcorr.shape)  # (correlation peak coordinates)
                match_result_coord = np.array([[corr_list[indx], y.get() + cropy, x.get() + cropx]])
                match_result_peak = np.array([[corr_list[indx], norm_xcorr[y,x].get()]])
                match_results_coord = np.append(match_results_coord, match_result_coord, axis=0)
                match_results_peak = np.append(match_results_peak, match_result_peak, axis=0)

            return match_results_coord, match_results_peak
