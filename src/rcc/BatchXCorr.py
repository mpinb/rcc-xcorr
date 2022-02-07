import concurrent.futures as cf
from multiprocessing.pool import ThreadPool

from .XCorrCpu import XCorrCpu
from .XCorrGpu import XCorrGpu
import numpy as np
import cupy as cp

from tqdm.auto import tqdm


# The index_correlations method prepends a correlation list
# with an extra column that is used to identify each correlation
def index_correlations(correlations):
    num_correlations = correlations.shape[0]
    corr_index = np.arange(num_correlations).reshape([num_correlations, 1])
    indexed_correlations = np.concatenate([corr_index, correlations], axis=1)
    return indexed_correlations


# The sorted correlations are prepended with an extra column
# with the initial order in which the correlations were sent
def sort_correlations(correlations):
    num_correlations = correlations.shape[0]
    orig_order = np.arange(num_correlations).reshape([num_correlations, 1])
    sorted_correlations = np.concatenate([orig_order, correlations], axis=1)
    sort_column = 1 # sorting correlations by image id
    sorted_correlations = sorted_correlations[np.argsort(sorted_correlations[:, sort_column])]
    return sorted_correlations


# NOTE: this function assumes a sorted correlations list as input
def group_correlations(sorted_correlations):
    correlations_size = len(sorted_correlations)
    sorted_correlations_image_column = 1
    correlation_images = sorted_correlations[:, sorted_correlations_image_column]
    (unique_images, unique_images_frequencies) = np.unique(correlation_images, return_counts=True)
    # removing the last element of the cumulative sum to avoid empty lists resulting from np.split
    # REF: https://numpy.org/doc/stable/reference/generated/numpy.split.html
    grouped_correlations = np.split(sorted_correlations, np.cumsum(unique_images_frequencies)[:-1])
    return grouped_correlations

# TODO:  A group correlations flag.  Sorting is only needed in the case of group correlations.
# TODO:  Indexing the correlations is done both for sorted and unsorted correlations.

class BatchXCorr:

    def __init__(self, images, templates, correlations,
                 normalize_input=False, crop_output=(0, 0), use_gpu=True):
        self.images = images
        self.templates = templates
        self.correlations = correlations
        self.normalize_input = normalize_input
        self.crop_output = crop_output
        self.use_gpu = use_gpu

    def perform_correlations(self):

        if self.use_gpu:
            xcorr = XCorrGpu(normalize_input=self.normalize_input, crop_output=self.crop_output)
        else:
            xcorr = XCorrCpu(normalize_input=self.normalize_input, crop_output=self.crop_output)

        futures = []
        with tqdm(total=len(self.correlations), delay=1) as progress:
            with cf.ThreadPoolExecutor(max_workers=4) as pool: # FIXME: dynamically set the amount of workers (can cause out of device memory errors)
                for corr_num, correlation in enumerate(self.correlations):
                    image_id, templ_id = correlation
                    future = pool.submit(xcorr.match_template, self.images[image_id], self.templates[templ_id], corr_num)
                    future.add_done_callback(lambda p: progress.update(1))
                    futures.append(future)

        # The results of correlations are kept in a numpy array internally
        batch_results_coord = np.empty((0, 2), int)
        batch_results_peak = np.empty((0, 1), float)

        for future in futures:
            x, y, peak = future.result()
            corr_result_coord = np.array([[x, y]])
            corr_result_peak = np.array([[peak]])
            batch_results_coord = np.append(batch_results_coord, corr_result_coord, axis=0)
            batch_results_peak = np.append(batch_results_peak, corr_result_peak, axis=0)

        # cleanup memory
        xcorr.cleanup()

        return batch_results_coord, batch_results_peak

    def perform_group_correlations(self):

        if self.use_gpu:
            xcorr = XCorrGpu(normalize_input=self.normalize_input)
        else:
            xcorr = XCorrCpu(normalize_input=self.normalize_input)

        # NOTE: sorting correlations optimize copies of data to gpu
        sorted_correlations = sort_correlations(self.correlations)
        # NOTE: group correlations based on the image column
        grouped_correlations = group_correlations(sorted_correlations)

        futures = []
        with tqdm(total=len(grouped_correlations), delay=1) as progress:
            with cf.ThreadPoolExecutor(max_workers=4) as pool: # FIXME: dynamically set the amount of workers
                for corr_list_num, correlation_group in enumerate(grouped_correlations):
                    corr_id_array, image_id_array, templ_id_array = np.split(correlation_group, 3, axis=1)
                    correlations_list = np.array(corr_id_array).flatten()
                    image_ids = np.array(image_id_array).flatten()
                    templ_ids = np.array(templ_id_array).flatten()
                    #Loading group image
                    group_image_id = image_ids[0]
                    group_image = self.images[group_image_id]
                    #Loading templates
                    templates_list = [self.templates[templ_id] for templ_id in templ_ids]
                    future = pool.submit(xcorr.match_template_array,
                                         group_image, templates_list, correlations_list, corr_list_num) # use corr_list_num to reorder the results
                    future.add_done_callback(lambda p: progress.update(1))
                    futures.append(future)

        # The results of correlations are kept in a numpy array internally
        batch_results_coord = np.empty((0, 3), int)
        batch_results_peak = np.empty((0, 2), float)

        for future in futures:
            group_coords, group_peaks = future.result()
            batch_results_coord = np.append(batch_results_coord, group_coords, axis=0)
            batch_results_peak = np.append(batch_results_peak, group_peaks, axis=0)

        corr_id_col = 0  # sorting batch correlation results by correlation id
        batch_results_coord = batch_results_coord[np.argsort(batch_results_coord[:, corr_id_col])]
        batch_results_peak = batch_results_peak[np.argsort(batch_results_peak[:, corr_id_col])]

        # cleanup memory
        xcorr.cleanup()

        # remove the correlation id column from batch results
        return batch_results_coord[:,1:], batch_results_peak[:, 1:] # removing the correlation_id column
