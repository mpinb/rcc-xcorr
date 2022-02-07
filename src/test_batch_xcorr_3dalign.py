# validate cross correlations for the 3d align case.
import resource
import time
import dill
import os
import numpy as np
import multiprocessing as mp

from rcc import BatchXCorr
import xcorr_util as xcu

#export_xcorr_comps_path = '/gpfs/soma_local/cne/watkins/xcorr_dump_macaque_3d_iorder3517'
#export_xcorr_comps_path = '/gpfs/soma_local/cne/watkins/xcorr_dump_macaque_w2_s1513_mfov29'
export_xcorr_comps_path = '/gpfs/soma_fs/scratch/valerio/xcorr_dump_macaque_3d_iorder3517'
plot_input_data = False
plot_statistics = False
normalize_inputs = False
group_correlations = False
limit_input_size = False
max_sample_size = 400 # value used when the limit input size flag is True
skip_correlations = True
num_skip_correlations = 0 # value used when skip correlations flag is True
crop_output = (221, 221) # use for the 3d align case
#crop_output = (0, 0) # use for the 2d align case
use_gpu = True

fn = os.path.join(export_xcorr_comps_path, 'comps.dill')
with open(fn, 'rb') as f: d = dill.load(f)

correlations = d['comps']
Cmax_test = d['Cmax']
Camax_test = d['Camax']

print(f'[BATCH_XCORR] Total correlations: {len(correlations)}')

if skip_correlations:
    correlations = correlations[num_skip_correlations:]
    Cmax_test = Cmax_test[num_skip_correlations:]
    Camax_test = Camax_test[num_skip_correlations:]
    print(f'[BATCH_XCORR] Skipping first {num_skip_correlations}. Remaining correlations: {len(correlations)}')

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
    #max_input_size = 200
    #input_size = max(len(image_files), max_input_size)
    #image_files = {k:v for count, (k,v) in enumerate(image_files.items()) if count < max_input_size}
    #templ_files = {k:v for count, (k,v) in enumerate(templ_files.items()) if count < max_input_size}
    #sample_correlations = input_size # assuming one-to-one image/template comparisons
else:
    sample_size = len(correlations)

# NOTE: Using dictionaries for testing. The final version will support a numpy array
# images = np.empty(NUM_IMAGES)
start_time = time.time()
print(f'[BATCH_XCORR] Thread pool size: {mp.cpu_count()}')
print(f'[BATCH_XCORR] Loading {len(image_files)} images.')
images = xcu.read_files_parallel_progress(image_files)
print(f'[BATCH_XCORR] Loading {len(templ_files)} templates.')
templates = xcu.read_files_parallel_progress(templ_files)
stop_time = time.time()
print(f'[BATCH_XCORR] Loading completed in {stop_time - start_time} seconds')

# Sampling memory use. (maximum resident set size in kilobytes)
usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f'[BATCH_XCORR] Current memory usage is {usage / 10 ** 3} MB')

print(f'[BATCH_XCORR] Using GPU: {use_gpu}')
print(f'[BATCH_XCORR] Grouping correlations (2D alignment optimization): {group_correlations}')

#if plot_statistics:
#    plot_sample_size = len(correlations)
#    xcu.plot_statistics(images, templates, correlations, plot_sample_size)

if plot_input_data:
    plot_sample_size = 5
    xcu.plot_input_data(images, templates, correlations, plot_sample_size)

print(f'Testing rcc-xcorr batch mode.')
start_time = time.time()
batch_correlations = BatchXCorr.BatchXCorr(images, templates, correlations[:sample_size],
                                           crop_output=crop_output, use_gpu=use_gpu)

if group_correlations:
    result_coords, result_peaks = batch_correlations.perform_group_correlations()
else:
    result_coords, result_peaks = batch_correlations.perform_correlations()

stop_time = time.time()
print(f"[BATCH_XCORR] elapsed time: {stop_time - start_time} seconds")

#del batch_correlations

#NOTE:  Using atol=1e-6 to compare computed vs. test correlation peak values
#REF:   https://stackoverflow.com/questions/57063555/numpy-allclose-compare-arrays-with-floating-points
print(f"Coordinates match: {np.allclose(result_coords, Camax_test[:sample_size])}")
print(f"Peak values match: {np.allclose(np.transpose(result_peaks), Cmax_test[:sample_size], atol=1e-5)}")

#print(np.transpose(result_peaks))
#print(Cmax_test[:sample_size])

#print(result_coords)
#print(Camax_test[:sample_size])

for correlation_indx, correlation, \
    test_peak, test_coord, \
    result_peak, result_coord \
        in zip(range(num_skip_correlations, num_skip_correlations+sample_size),
                  correlations[:sample_size],
                  Cmax_test[:sample_size],
                  Camax_test[:sample_size],
                  result_peaks,
                  result_coords):
    if not(np.allclose(result_coord, test_coord) and np.isclose(result_peak, test_peak, atol=1e-5)):
        print(f'[MISMATCH CORR] CORR_ID: {correlation_indx}, CORRELATION: {correlation}')
        print(f'[MISMATCH COORD] RCOORD:{result_coord} TCOORD:{test_coord}')
        print(f'[MISMATCH PEAK] RMAX: {result_peak} TMAX:{test_peak}')
        xcu.plot_xcorr(correlation, images, templates,
                       crop_output=crop_output,
                       expected_max=test_peak,
                       expected_max_coord=test_coord)
