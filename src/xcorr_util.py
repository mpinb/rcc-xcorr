import os
import re
import tqdm
import tifffile
import numpy as np
import multiprocessing as mp
import concurrent.futures as cf
from matplotlib import pyplot as plt
from multiprocessing.pool import ThreadPool


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
def read_files_parallel_progress(files, num_procs=mp.cpu_count()):
    with ThreadPool(num_procs) as pool:
        return {file_id:file_data for file_id, file_data in
                zip(files.keys(),
                    tqdm.tqdm(pool.imap(tifffile.imread, files.values()), total=len(files)))}
