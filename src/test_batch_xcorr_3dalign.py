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
use_gpu = True

fn = os.path.join(export_xcorr_comps_path, 'comps.dill')
with open(fn, 'rb') as f: d = dill.load(f)

correlations = d['comps']
Cmax_test = d['Cmax']
Camax_test = d['Camax']

# Gathering the file names of images and templates
image_files = xcu.search_files(export_xcorr_comps_path, r'image([0-9]+)\.tif')
templ_files = xcu.search_files(export_xcorr_comps_path, r'templ([0-9]+)\.tif')

# NOTE: Using dictionaries for testing. The final version will support a numpy array
# images = np.empty(NUM_IMAGES)
print(f'[BATCH_XCORR] Reading files using process pool of size: {mp.cpu_count()}')
start_time = time.time()
images = xcu.read_files_parallel(image_files)
templates = xcu.read_files_parallel(templ_files)
stop_time = time.time()
print(f'[BATCH_XCORR] Elapsed time reading files: {stop_time - start_time} seconds')

# Sampling memory use. (maximum resident set size in kilobytes)
usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f'[BATCH_XCORR] Current memory usage is {usage / 10 ** 3} MB')

print(f'[BATCH_XCORR] Using GPU: {use_gpu}')
print(f'[BATCH_XCORR] Grouping correlations (2D alignment): {group_correlations}')
print(f'[BATCH_XCORR] Total read images: {len(images)}')
print(f'[BATCH_XCORR] Total read templates: {len(templates)}')

#if plot_statistics:
#    plot_sample_size = len(correlations)
#    xcu.plot_statistics(images, templates, correlations, plot_sample_size)

if plot_input_data:
    plot_sample_size = 5
    xcu.plot_input_data(images, templates, correlations, plot_sample_size)

print(f'Testing rcc-xcorr batch mode.')
start_time = time.time()
#sample_correlations = 5
sample_correlations = len(correlations)
batch_correlations = BatchXCorr.BatchXCorr(images, templates, correlations[:sample_correlations], use_gpu=use_gpu)

if group_correlations:
    result_coords, result_peaks = batch_correlations.perform_group_correlations()
else:
    result_coords, result_peaks = batch_correlations.perform_correlations()

stop_time = time.time()
print(f"[BATCH_XCORR] elapsed time: {stop_time - start_time} seconds")

#NOTE:  Using atol=1e-6 to compare computed vs. test correlation peak values
#REF:   https://stackoverflow.com/questions/57063555/numpy-allclose-compare-arrays-with-floating-points
print(f"Coordinates match: {np.allclose(result_coords, Camax_test[:sample_correlations])}")
print(f"Peak values match: {np.allclose(np.transpose(result_peaks), Cmax_test[:sample_correlations], atol=1e-6)}")

#print(np.transpose(result_peaks))
#print(Cmax_test[:sample_correlations])

#print(result_coords)
#print(Camax_test[:sample_correlations])
