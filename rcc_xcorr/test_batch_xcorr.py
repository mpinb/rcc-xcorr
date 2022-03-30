# validate cross correlations for a single set of mfov comparisons
#   as dumped from mfov using the export_xcorr_comps_path hack.
import time
import tifffile
import dill
import os
import re
import numpy as np

from xcorr import BatchXCorr
from rcc_xcorr.xcorr import XCorrUtil as xcu

export_xcorr_comps_path = '/gpfs/soma_local/cne/watkins/xcorr_dump_macaque_w2_s1513_mfov29'
#export_xcorr_comps_path = '/gpfs/soma_fs/scratch/valerio/xcorr_dump_macaque_w2_s1513_mfov29'

plot_input_data = False
plot_statistics = True
normalize_inputs = False
group_correlations = True
use_gpu = True
disable_pbar = False

fn = os.path.join(export_xcorr_comps_path, 'comps.dill')
with open(fn, 'rb') as f: d = dill.load(f)

correlations = d['comps']
Cmax_test = d['Cmax']
Camax_test = d['Camax']

# NOTE: adding reference correlation used for debugging
# image0000.tiff, templ0000.tiff
correlations = np.vstack ((np.array([0, 0]), correlations))
Cmax_test = np.append(np.array([1.000000]), Cmax_test)
Camax_test = np.vstack((np.array([364, 749]), Camax_test)) # (y,x)

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

print(f'[BATCH_XCORR] Using GPU: {use_gpu}')
print(f'[BATCH_XCORR] Grouping correlations (2D alignment): {group_correlations}')
print(f'[BATCH_XCORR] Total read images: {len(images)}')
print(f'[BATCH_XCORR] Total read templates: {len(templates)}')

if plot_statistics:
    plot_sample_size = len(correlations)
    xcu.plot_statistics(images, templates, correlations, plot_sample_size)

if plot_input_data:
    plot_sample_size = 5
    xcu.plot_input_data(images, templates, correlations, plot_sample_size)

print(f'Testing rcc-xcorr batch mode.')
start_time = time.time()
#sample_correlations = 5
sample_correlations = len(correlations)
batch_correlations = BatchXCorr.BatchXCorr(images, templates, correlations[:sample_correlations],
                                           use_gpu=use_gpu, group_correlations=group_correlations,
                                           disable_pbar=disable_pbar)
result_coords, result_peaks = batch_correlations.execute_batch()

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
