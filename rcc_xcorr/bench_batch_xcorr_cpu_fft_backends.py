# plot performance of CPU based cross-correlations
# using emalign test data
import dill
import os
import sys
import time
import resource
import perfplot
import logging

import scipy
#import pyfftw
#import mkl_fft

from pyfftw.interfaces import scipy_fft as pyfftw_backend
from mkl_fft import _scipy_fft_backend as mkl_backend

import multiprocessing as mp

import numpy as np

from rcc_xcorr.xcorr import BatchXCorr
from rcc_xcorr.xcorr import XCorrUtil as xcu

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setStream(sys.stdout)
logger.addHandler(handler)

# The new scipy.fft subpackage should be extended to add a backend system with support for PyFFTW and mkl-fft.
def scipy_fft(images, templates, correlations, crop_output, use_gpu, num_workers):
    print(f'[BATCH_XCORR] using scipy backend (num_workers={num_workers})')
    scipy.fft.set_workers(num_workers)
    with scipy.fft.set_backend('scipy', only=True):
        BatchXCorr.BatchXCorr(  images, templates, correlations, crop_output=crop_output,
                                use_gpu=use_gpu, num_workers=1).execute_batch()

# The full list of scipy fft backends
#REF: https://github.com/scipy/scipy/issues/12509#issuecomment-656946077
def pyfftw_fft(images, templates, correlations, crop_output, use_gpu, num_workers):
    print(f'[BATCH_XCORR] using pyfftw backend (num_workers={num_workers})')
    scipy.fft.set_workers(num_workers)
    with scipy.fft.set_backend(pyfftw_backend, only=True):
        BatchXCorr.BatchXCorr(  images, templates, correlations, crop_output=crop_output,
                                use_gpu=use_gpu, num_workers=1).execute_batch()


#REF: https://pypi.org/project/mkl-fft/
#Install conda install -c intel mkl_fft  OR pip install mkl-fft
# conda install -c conda-forge mkl_fft
# conda install -c conda-forge mkl-service
#REF: https://github.com/MattKleinsmith/pbt/issues/3
def mkl_fft(images, templates, correlations, crop_output, use_gpu, num_workers):
    print(f'[BATCH_XCORR] using mkl_fft backend (num_workers={num_workers})')
    scipy.fft.set_workers(num_workers)
    with scipy.fft.set_backend(mkl_backend, only=True):
        BatchXCorr.BatchXCorr(  images, templates, correlations, crop_output=crop_output,
                                use_gpu=use_gpu, num_workers=1).execute_batch()


if __name__ == '__main__':

    benchmark_plot_filename = 'benchmark_xcorr_cpu_fft_backends.svg'
    export_xcorr_comps_path = '/gpfs/soma_local/cne/watkins/xcorr_dump_macaque_3d_iorder3517'
    #export_xcorr_comps_path = '/gpfs/soma_fs/scratch/valerio/xcorr_dump_macaque_3d_iorder3517'

    normalize_inputs = False
    limit_input_size = True
    # max_sample_size = 300  # value used when the limit input size flag is True
    max_sample_size = 100  # value used when the limit input size flag is True
    crop_output = (221, 221)  # use for the 3d align case

    fn = os.path.join(export_xcorr_comps_path, 'comps.dill')
    with open(fn, 'rb') as f: d = dill.load(f)

    correlations = d['comps']
    Cmax_test = d['Cmax']
    Camax_test = d['Camax']

    # Gathering the file names of images and templates
    image_files = xcu.search_files(export_xcorr_comps_path, r'image([0-9]+)\.tif')
    templ_files = xcu.search_files(export_xcorr_comps_path, r'templ([0-9]+)\.tif')

    # To test in memory constrained environments
    # The approach is not robust. It relies on the dictionary to preserve the read order
    if limit_input_size:
        sample_size = min(len(correlations), max_sample_size)
        print(f'[BATCH_XCORR] Limiting input size. Sample size: {sample_size}')
        image_set, template_set = xcu.sampled_correlations_input(correlations, sample_size)
        image_files = {k:image_files[k] for k in image_set}
        templ_files = {k:templ_files[k] for k in template_set}
    else:
        sample_size = len(correlations)

    # NOTE: using dictionaries as a proxy for numpy arrays
    # images = np.empty(NUM_IMAGES)
    start_time = time.time()
    print(f'[BATCH_XCORR] Loading {len(image_files)} images.')
    images = xcu.read_files_parallel_progress(image_files)
    print(f'[BATCH_XCORR] Loading {len(templ_files)} templates.')
    templates = xcu.read_files_parallel_progress(templ_files)
    stop_time = time.time()
    print(f'[BATCH_XCORR] Loading completed in {stop_time - start_time} seconds')

    # Sampling memory use. (maximum resident set size in kilobytes)
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f'[BATCH_XCORR] Current memory usage is {usage / 10 ** 3} MB')

    print(f'[BATCH_XCORR] benchmarking rcc-xcorr CPU scaling.')

    print(f'[BATCH_XCORR] sample_size: {sample_size}')
    n_range = np.linspace(30, sample_size, num=8, dtype=int).tolist()
    print(f'[BATCH_XCORR] n_range: {n_range}')

    use_gpu = False
    num_workers = 8 # mp.cpu_count()
    fft_callbacks = [scipy_fft, pyfftw_fft, mkl_fft]
    enable_fftw = [False, True]
    labels = ["XCorrCpu (scipy)", "XCorrCpu (pyfftw)", "XCorrCpu (mkl_fft)"]

    kernels = list(map(
        lambda fft_callback: (
            lambda correlations: fft_callback(images, templates, correlations, crop_output, use_gpu, num_workers))
        , fft_callbacks))

    bench_scaling_out =perfplot.bench(
        setup=lambda n: correlations[np.random.choice(sample_size, size=n, replace=True)],
        kernels=kernels,
        labels=labels,
        n_range=n_range, # sample points
        xlabel="total correlations",
        # More optional arguments with their default values:
        # logx=False,  # set to True or False to force scaling
        # logy="auto",
        equality_check=None,  # set to None to disable "correctness" assertion
        # show_progress=True,
        # target_time_per_measurement=1.0,
        # max_time=None,  # maximum time per measurement
        # time_unit="s",  # set to one of ("auto", "s", "ms", "us", or "ns") to force plot units
        # relative_to=1,  # plot the timings relative to one of the measurements
        # flops=lambda n: 3*n,  # FLOPS plots
    )
    bench_scaling_out.show()
    bench_scaling_out.save(benchmark_plot_filename, transparent=True, bbox_inches="tight")
