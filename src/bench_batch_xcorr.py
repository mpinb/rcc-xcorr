# plot performance of both GPU and CPU based cross-correlations
# using emalign test data
import tifffile
import dill
import os
import re

import numpy as np
import perfplot

from rcc import BatchXCorr


benchmark_plot_filename = 'benchmark_xcorr.png'
export_xcorr_comps_path = '/gpfs/soma_fs/cne/watkins/xcorr_dump_macaque_w2_s1513_mfov29'
normalize_inputs = False
use_gpu = True

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


print(f'[BATCH_XCORR] benchmarking {"GPU" if use_gpu else "CPU"} correlation kernels.')
correlations_size = len(correlations)
#correlations_size = 20
print(f'[BATCH_XCORR] sample_correlations_size: {correlations_size}')
print(f'[BATCH_XCORR] n_range: {np.linspace(10, correlations_size, num=10, dtype=int).tolist()}')

labels_cpu = ["XCorrCPU", "XCorrCPU(group)"]
labels_gpu = ["XCorrGPU", "XCorrGPU(group)"]

perfplot.live(
    setup= lambda n: correlations[np.random.choice(correlations_size, size=n, replace=True)],
    kernels=[
        lambda correlations_sample: BatchXCorr.BatchXCorr(images, templates, correlations_sample,
                                                          use_gpu=use_gpu).perform_correlations(),
        lambda correlations_sample: BatchXCorr.BatchXCorr(images, templates, correlations_sample,
                                                          use_gpu=use_gpu).perform_group_correlations()
    ],
    labels=labels_gpu if use_gpu else labels_cpu,
    #n_range=[2 ** k for k in range(1,11)],
    n_range=np.linspace(10, correlations_size, num=10, dtype=int).tolist(), # use 20 sample points
    xlabel="len(correlations_sample)",
    # More optional arguments with their default values:
    logx=False,  # set to True or False to force scaling
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

