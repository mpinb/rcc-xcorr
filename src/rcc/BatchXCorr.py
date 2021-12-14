from .XCorrCpu import XCorrCpu
from .XCorrGpu import XCorrGpu
import numpy as np
import cupy as cp

from tqdm import tqdm

# The sorted correlations are prepended with an extra column
# with the initial order in which the correlations were sent
def sort_correlations(correlations):
    num_correlations = correlations.shape[0]
    orig_order = np.arange(num_correlations).reshape([num_correlations, 1])
    sorted_correlations = np.concatenate([orig_order, correlations], axis=1)
    sort_column = 1 # sorting correlations by image id
    sorted_correlations = sorted_correlations[np.argsort(sorted_correlations[:, sort_column])]
    return sorted_correlations

class BatchXCorr:

    def __init__(self, images, templates, correlations, normalize_output=True, normalize_input=False, use_gpu=True):
        self.images = images
        self.templates = templates
        self.correlations = correlations
        self.normalize_output = normalize_output
        self.normalize_input = normalize_input
        self.use_gpu = use_gpu

    def perform_correlations(self):

        # NOTE: sorting correlations optimize copies of data to gpu
        sorted_correlations = sort_correlations(self.correlations)
        #print(sorted_correlations)

        # TODO: remove unused images & templates if not found in correlations

        if self.use_gpu:
            xcorr = XCorrGpu(True, False)
        else:
            xcorr = XCorrCpu(True, False)

        # The results of correlations are kept in a numpy array internally
        batch_results_coord = np.empty((0, 3), int)
        batch_results_peak = np.empty((0, 2), float)

        num_correlations = len(sorted_correlations)
        for indx in tqdm(range(num_correlations)):
            correlation = sorted_correlations[indx]
            corr_id, image_id, templ_id = correlation
            #print(f"perform correlation: {corr_id} image: {image_id} templ: {templ_id}")
            x, y, peak = xcorr.match_template(self.images[image_id], self.templates[templ_id])
            corr_result_coord = np.array([[corr_id, x, y]])
            corr_result_peak = np.array([[corr_id, peak]])
            batch_results_coord = np.append(batch_results_coord, corr_result_coord, axis=0)
            batch_results_peak = np.append(batch_results_peak, corr_result_peak, axis=0)

        corr_id_col = 0  # sorting batch correlation results by correlation id
        batch_results_coord = batch_results_coord[np.argsort(batch_results_coord[:, corr_id_col])]
        batch_results_peak = batch_results_peak[np.argsort(batch_results_peak[:, corr_id_col])]

        # remove the correlation id column from batch results
        return batch_results_coord[:,1:], batch_results_peak[:, 1:] # removing the correlation_id column

    def perform_sorted_correlations_gpu(self, sorted_correlations):

        # Copy images to gpu memory
        # FIXME: this copy is not correct when using Python dictionaries
        images_data = list(self.images.values())
        images_gpu = cp.array(images_data)

        # Copy templates to gpu memory
        # FIXME: this copy is not correct when using Python dictionaries
        templates_data = list(self.templates.values())
        templates_gpu = cp.array(templates_data)

        # The results of correlations are kept in a numpy array internally
        batch_results_coord = np.empty((0, 3), int)
        batch_results_peak = np.empty((0, 2), float)

        num_correlations = len(sorted_correlations)
        for indx in tqdm(range(num_correlations)): #for correlation in sorted_correlations:
            correlation = sorted_correlations[indx]
            corr_id, image_id, templ_id = correlation
            #print(f"perform correlation: {corr_id} image: {image_id} templ: {templ_id}")
            xcorr_gpu = XCorrGpu(True, False)
            x, y, peak = xcorr_gpu.match_template(images_gpu[image_id], templates_gpu[templ_id])
            corr_result_coord = np.array([[corr_id, x, y]])
            corr_result_peak = np.array([[corr_id, peak]])
            batch_results_coord = np.append(batch_results_coord, corr_result_coord, axis=0)
            batch_results_peak = np.append(batch_results_peak, corr_result_peak, axis=0)

        corr_id_col = 0  # sorting batch correlation results by correlation id
        batch_results_coord = batch_results_coord[np.argsort(batch_results_coord[:, corr_id_col])]
        batch_results_peak = batch_results_peak[np.argsort(batch_results_peak[:, corr_id_col])]

        # remove the correlation id column from batch results
        return batch_results_coord[:,1:], batch_results_peak[:, 1:] # removing the correlation_id column
