# plot performance of both GPU and CPU based cross-correlations
# using emalign test data
import dill
import os
import sys
import time
import resource
import perfplot
import logging

import multiprocessing as mp

import cupy as cp
import numpy as np

from xcorr import BatchXCorr
from rcc_xcorr.xcorr import XCorrUtil as xcu

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setStream(sys.stdout)
logger.addHandler(handler)


# using a wrapper to run the BatchXCorr in a separate process.
# NOTE: the GPU has to run in a separate process to free memory.
def wrapper_batch_xcorr(images, templates, correlations, crop_output, use_gpu, num_gpu, num_worker):
    logger.info(f'[PID: {os.getpid()}] total correlations: {len(correlations)}')
    logger.info(f'[PID: {os.getpid()}] use_gpu: {use_gpu}, num_gpu: {num_gpu}, num_worker: {num_worker}')
    BatchXCorr.BatchXCorr(
        images, templates, correlations, crop_output=crop_output,
        use_gpu=use_gpu, num_gpus=num_gpu, num_workers=num_worker).execute_batch()


# run the wrapper function in a separate process
def run_proc(wrapper_func, *args):
    p = mp.Process(target=wrapper_func,args=args)
    p.start()
    p.join()


if __name__ == '__main__':

    # REF1: https://github.com/pytorch/pytorch/issues/3492
    # REF2: https://stackoverflow.com/questions/54808148/cupy-get-error-in-multithread-pool-if-gpu-already-used
    # REF3: https://github.com/explosion/spaCy/issues/5507
    mp.set_start_method('spawn')

    benchmark_plot_filename = 'benchmark_xcorr_gpu_scaling.svg'
    #export_xcorr_comps_path = '/gpfs/soma_local/cne/watkins/xcorr_dump_macaque_3d_iorder3517'
    export_xcorr_comps_path = '/gpfs/soma_fs/scratch/valerio/xcorr_dump_macaque_3d_iorder3517'

    normalize_inputs = False
    limit_input_size = True
    max_sample_size = 300  # value used when the limit input size flag is True
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

    print(f'[BATCH_XCORR] benchmarking rcc-xcorr GPU scaling.')

    print(f'[BATCH_XCORR] sample_size: {sample_size}')
    n_range = np.linspace(30, sample_size, num=8, dtype=int).tolist()
    print(f'[BATCH_XCORR] n_range: {n_range}')

    #use_gpu = True
    max_gpus = cp.cuda.runtime.getDeviceCount()
    print(f'[BATCH_XCORR] max_gpus: {max_gpus}')
    num_gpus = np.linspace(0, max_gpus, num=max_gpus+1, dtype=int).tolist() # Benchmark CPU and GPU
    # num_gpus = np.linspace(1, max_gpus, num=max_gpus, dtype=int).tolist() # NOTE: Benchmark only GPU

    # NOTE: Adjust number of workers per GPU according to the running environment (e.g: SOMA = 4)
    workers_per_gpu = 4
    num_workers = [workers_per_gpu * num_gpu if num_gpu else mp.cpu_count() for num_gpu in num_gpus]
    labels = [f"XCorr(num_gpus={gpu},num_workers={worker})" for gpu, worker in zip(num_gpus, num_workers)]
    use_gpus = [True if gpu else False for gpu in num_gpus]

    kernels = list(map(
        lambda use_gpu, num_gpu, num_worker: (
            lambda correlations: \
                run_proc(wrapper_batch_xcorr, images, templates, correlations, crop_output, use_gpu, num_gpu, num_worker)
        )
        , use_gpus, num_gpus, num_workers))

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

