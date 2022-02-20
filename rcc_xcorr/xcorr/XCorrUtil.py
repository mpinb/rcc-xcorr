import os
import re
import logging
import tifffile
import cupy as cp
import numpy as np
import multiprocessing as mp
import concurrent.futures as cf

from .XCorrCpu import XCorrCpu
from .XCorrGpu import XCorrGpu

from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from multiprocessing.pool import ThreadPool


# REF: https://stackoverflow.com/a/38739634
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def sampled_correlations_input(correlations, sample_size):
    image_set = set()
    template_set = set()
    for correlation in correlations[:sample_size]:
        image_id, templ_id = correlation
        image_set.add(image_id)
        template_set.add(templ_id)
    return image_set, template_set


def plot_input_data(images, templates, correlations, sample_size):
    total_correlations = correlations.shape[0]
    for c in range(min(sample_size, total_correlations)):
        # load the inputs and correlate
        image_id, templ_id = correlations[c, :]
        print('Comparing img {} to tpl {}'.format(image_id, templ_id))
        plt.figure(c)
        plt.subplot(1, 2, 1)
        plt.title('image')
        plt.imshow(images[image_id], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('template')
        plt.imshow(templates[templ_id], cmap='gray')
        plt.show()


def plot_xcorr(correlation, images, templates, crop_output, expected_max, expected_max_coord):
    print(f'correlation: {correlation}')
    image_id, templ_id = correlation
    print(f'Plotting correlation between image: {image_id} and template: {templ_id}')
    image = images[image_id]
    template = templates[templ_id]
    xcorr_cpu = XCorrCpu(cache_correlation=True, crop_output=crop_output)
    xcorr_gpu = XCorrGpu(cache_correlation=True, crop_output=crop_output)
    ym_cpu, xm_cpu, max_cpu = xcorr_cpu.match_template(image, template)
    ym_gpu, xm_gpu, max_gpu = xcorr_gpu.match_template(image, template)
    correlation_cpu = xcorr_cpu.get_correlation()
    correlation_gpu = cp.asnumpy(xcorr_gpu.get_correlation())
    print(f'Image shape: {image.shape} Correlation shape: {correlation_cpu.shape}')
    print(f'Image type: {image.dtype} Correlation type: {correlation_cpu.dtype}')
    f, axes = plt.subplots(2, 2)
    f.suptitle(f'Expected Correlation max: {expected_max:.6f} (y,x): {expected_max_coord}', y=0.04)
    axes[0,0].set_title(f'Image\nid: {image_id}')
    axes[0,0].imshow(image, cmap='gray')
    axes[0,0].plot(expected_max_coord[1], expected_max_coord[0], #NOTE: ex_max_coord(y,x)
             color='green', marker='o', markersize=12, fillstyle='none', linewidth=2)
    axes[0,1].set_title(f'Template\nid: {templ_id}')
    axes[0,1].imshow(template, cmap='gray')
    axes[1,0].set_title(f'XCorr CPU\nmax: {max_cpu:.6f}\n(y,x):({ym_cpu},{xm_cpu})')
    axes[1,0].imshow(correlation_cpu, cmap='gray')
    axes[1,0].plot(xm_cpu, ym_cpu,
             color='green', marker='o', markersize=12, fillstyle='none', linewidth=2)
    axes[1,1].set_title(f'XCorr GPU\nmax: {max_gpu:.6f}\n(y,x):({ym_gpu},{xm_gpu})')
    axes[1,1].imshow(correlation_gpu, cmap='gray')
    axes[1,1].plot(xm_gpu, ym_gpu,
             color='green', marker='o', markersize=12, fillstyle='none', linewidth=2)
    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    plt.show()


# Plot correlations statistics using a cumulative histogram
# REF: https://matplotlib.org/stable/gallery/statistics/histogram_cumulative.html
def plot_statistics(images, templates, correlations, sample_size):
    sample_size = min(len(correlations), sample_size)
    images_sample = correlations[:sample_size, 0]
    (used_images, images_counts) = np.unique(images_sample, return_counts=True)
    templates_sample = correlations[:sample_size, 1]
    (used_templates, templates_counts) = np.unique(templates_sample, return_counts=True)

    total_read_images = len(images)
    total_read_templates = len(templates)

    total_used_images = len(used_images)
    total_used_templates = len(used_templates)

    plt.figure(sample_size)
    plt.subplot(1, 2, 1)
    plt.title(f'comp image stats\n(used images: {total_used_images})\n(read images: {total_read_images})')
    plt.hist(images_sample, total_used_images, density=False, histtype='step', cumulative=True, label="Images")
    plt.ylabel('total correlations')
    plt.xlabel('image id')
    plt.subplot(1, 2, 2)
    plt.title(f'comp templ stats\n(used templates: {total_used_templates})\n(read templates: {total_read_templates})')
    plt.hist(templates_sample, total_used_templates, density=False, histtype='step', cumulative=True, label="Images")
    plt.xlabel('template id')

    plt.show()


# return a dictionary of files matching filename regex pattern
# the regex also defines the key to use for the dictionary
# The filename_regex has the form: r'FILE_PREFIX(FILE_KEY)\.EXT'
# Example:  r'image([0-9]+)\.tif'
def search_files(file_path, filename_regex):
    files = {}
    for f in os.listdir(file_path):
        file_match = re.match(filename_regex, f)
        if file_match:
            file_id = int(file_match.group(1))
            file_name = os.path.join(file_path, f)
            files[file_id] = file_name
    return files

#import psutil
#psutil.cpu_count(logical = True)
# This operation is IO bound therefore using a ThreadPool executor
def read_files_parallel(files, num_procs=mp.cpu_count()):
    with cf.ThreadPoolExecutor(num_procs) as pool:
        return {file_id:file_data for file_id, file_data in
                zip(files.keys(),
                    pool.map(tifffile.imread, files.values()))}


# This one works exactly the same as read_files_parallel but uses
# tqdm package to show the progress of the parallel files being read
def read_files_parallel_progress(files, num_procs=4):
    futures = []
    with tqdm(total=len(files), position=0, leave=True, delay=2) as progress:
        with cf.ThreadPoolExecutor(max_workers=num_procs) as pool:
            for file_id in files.keys():
                future = pool.submit(tifffile.imread, files[file_id])
                future.add_done_callback(lambda p: progress.update(1))
                futures.append(future)

    return {file_id: future.result() for file_id, future in
            zip(files.keys(), futures)}
