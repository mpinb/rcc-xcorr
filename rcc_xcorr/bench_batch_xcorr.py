# plot performance of both GPU and CPU based cross-correlations
# using emalign test data
import tifffile
import dill
import os
import re
import sys

import multiprocessing as mp

import numpy as np
import perfplot
import logging

from xcorr import BatchXCorr

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

    benchmark_plot_filename = 'benchmark_xcorr.svg'
    #export_xcorr_comps_path = '/gpfs/soma_fs/cne/watkins/xcorr_dump_macaque_w2_s1513_mfov29'
    export_xcorr_comps_path = '/gpfs/soma_local/cne/watkins/xcorr_dump_macaque_w2_s1513_mfov29'
    normalize_inputs = False

    fn = os.path.join(export_xcorr_comps_path, 'comps.dill')
    with open(fn, 'rb') as f: d = dill.load(f)

    correlations = d['comps']
    Cmax_test = d['Cmax']
    Camax_test = d['Camax']

    #NOTE: using dictionary as a proxy but it would be instead a numpy array
    # images = np.empty(NUM_IMAGES)
    images = {}
    templates = {}

    for f in os.listdir(export_xcorr_comps_path):
        image_match = re.match(r'image([0-9]+)\.tif', f)
        if image_match:
            image_id = int(image_match.group(1))
            images[image_id] = tifffile.imread(os.path.join(export_xcorr_comps_path, f))
            continue
        templ_match = re.match(r'templ([0-9]+)\.tif', f)
        if templ_match:
            templ_id = int(templ_match.group(1))
            templates[templ_id] = tifffile.imread(os.path.join(export_xcorr_comps_path, f))

    print(f'[BATCH_XCORR] Total read images: {len(images)}')
    print(f'[BATCH_XCORR] Total read templates: {len(templates)}')


    print(f'[BATCH_XCORR] benchmarking GPU vs CPU correlation kernels.')
    #correlations_size = len(correlations)
    correlations_size = 100
    print(f'[BATCH_XCORR] sample_correlations_size: {correlations_size}')
    print(f'[BATCH_XCORR] n_range: {np.linspace(10, correlations_size, num=10, dtype=int).tolist()}')

    labels = ["XCorrCPU", "XCorrCPU(group)", "XCorrGPU", "XCorrGPU(group)"]
    use_gpus = [False, False, True, True]
    num_gpus = [0, 0, 1, 1]
    use_grouping = [False, True, False, True]
    # NOTE: Adjust number of workers per GPU according to the running environment (eg.: SOMA = 4 wks)
    workers_per_gpu = 3
    num_workers = [workers_per_gpu if use_gpu else mp.cpu_count() for use_gpu in use_gpus]

    kernels = list(map(
        lambda use_gpu, num_gpu, num_worker: (
            lambda correlations: \
                run_proc(wrapper_batch_xcorr,
                         images, templates, correlations,
                         (0, 0), # crop_output
                         use_gpu,
                         num_gpu, # num_gpu
                         num_worker)
        ), use_gpus, num_gpus, num_workers))

    bench_cpu_vs_gpu_out = perfplot.bench(
        setup=lambda n: correlations[np.random.choice(correlations_size, size=n, replace=True)],
        kernels=kernels,
        # kernels=[
        #     lambda correlations_sample: \
        #         BatchXCorr.BatchXCorr(images, templates, correlations_sample,
        #                               group_correlations=False, use_gpu=False).execute_batch(),
        #     lambda correlations_sample: \
        #         BatchXCorr.BatchXCorr(images, templates, correlations_sample,
        #                               group_correlations=True, use_gpu=False).execute_batch(),
        #     lambda correlations_sample: \
        #         BatchXCorr.BatchXCorr(images, templates, correlations_sample,
        #                               group_correlations=False, use_gpu=True).execute_batch(),
        #     lambda correlations_sample: \
        #         BatchXCorr.BatchXCorr(images, templates, correlations_sample,
        #                               group_correlations=True, use_gpu=True).execute_batch()
        # ],
        labels=labels,
        #n_range=[2 ** k for k in range(1,11)],
        n_range=np.linspace(10, correlations_size, num=10, dtype=int).tolist(), # use 20 sample points
        xlabel="total correlations",
        # More optional arguments with their default values:
        # logx=False,  # set to True or False to force scaling
        # logy="auto",
        equality_check=None,
        # equality_check=np.allclose,  # set to None to disable "correctness" assertion
        # show_progress=True,
        # target_time_per_measurement=1.0,
        # max_time=None,  # maximum time per measurement
        # time_unit="s",  # set to one of ("auto", "s", "ms", "us", or "ns") to force plot units
        # relative_to=1,  # plot the timings relative to one of the measurements
        # flops=lambda n: 3*n,  # FLOPS plots
    )
    bench_cpu_vs_gpu_out.show()
    bench_cpu_vs_gpu_out.save(benchmark_plot_filename, transparent=True, bbox_inches="tight")
