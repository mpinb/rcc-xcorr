# validate cross correlations for a single set of mfov comparisons
#   as dumped from mfov using the export_xcorr_comps_path hack.
import random
import time
import tifffile
import dill
import os
import re
import numpy as np

from rcc_xcorr.xcorr import BatchXCorr

try:
    dict(globals())['export_xcorr_comps_path']
except KeyError:
    #export_xcorr_comps_path = '/gpfs/soma_fs/cne/watkins/xcorr_dump_macaque_w2_s1513_mfov29'
    export_xcorr_comps_path = '/gpfs/soma_local/cne/watkins/xcorr_dump_macaque_w2_s1513_mfov29'
    #export_xcorr_comps_path = '/gpfs/soma_fs/scratch/valerio/xcorr_dump_macaque_w2_s1513_mfov29'

print(f'[BATCH_XCORR] export_xcorr_comps_path: {export_xcorr_comps_path}')

normalize_inputs = False
group_correlations = True
use_gpu = True
disable_pbar = False

fn = os.path.join(export_xcorr_comps_path, 'comps.dill')
with open(fn, 'rb') as f: d = dill.load(f)

correlations = d['comps']
Cmax_test = d['Cmax']
Camax_test = d['Camax']

#NOTE: using dictionary as a proxy but it would be instead a numpy array
# images = np.empty(NUM_IMAGES)

max_image_id = max_templ_id = -1
for f in os.listdir(export_xcorr_comps_path):
    image_match = re.match(r'image([0-9]+)\.tif', f)
    if image_match:
        image_id = int(image_match.group(1))
        #images[image_id] = tifffile.imread(os.path.join(export_xcorr_comps_path, f))
        if image_id > max_image_id: max_image_id = image_id
        continue
    templ_match = re.match(r'templ([0-9]+)\.tif', f)
    if templ_match:
        templ_id = int(templ_match.group(1))
        if templ_id > max_templ_id: max_templ_id = templ_id
        #templates[templ_id] = tifffile.imread(os.path.join(export_xcorr_comps_path, f))
images = [None]*(max_image_id+1)
templates = [None]*(max_templ_id+1)
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

print(f'Testing rcc-xcorr batch mode.')
start_time = time.time()
sample_correlations = 300
#sample_correlations = len(correlations)
#nrepeats = 70
nrepeats = 10
for i in range(nrepeats):
    print('repeat {} of {}'.format(i+1,nrepeats))
    num_devices = random.randint(1, 4)
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in range(0, num_devices)])
    batch_correlations = BatchXCorr.BatchXCorr(images, templates, correlations[:sample_correlations],
                                               use_gpu=use_gpu, group_correlations=group_correlations,
                                               num_gpus=num_devices, disable_pbar=disable_pbar)
    result_coords, result_peaks = batch_correlations.execute_batch()

    stop_time = time.time()
    print(f"[BATCH_XCORR] elapsed time: {stop_time - start_time} seconds")

    #NOTE:  Using atol=1e-6 to compare computed vs. test correlation peak values
    #REF:   https://stackoverflow.com/questions/57063555/numpy-allclose-compare-arrays-with-floating-points
    print(f"Coordinates match: {np.allclose(result_coords, Camax_test[:sample_correlations])}")
    print(f"Peak values match: {np.allclose(np.transpose(result_peaks), Cmax_test[:sample_correlations], atol=1e-6)}")

