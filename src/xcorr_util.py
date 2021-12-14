import numpy as np
from matplotlib import pyplot as plt


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

#def check_xcorr_results_equality(xcorr_results_gpu, xcorr_results_cpu):
#    print(f'[XCU_UTIL] xcorr_results: {xcorr_results_gpu}')
    #result_coords, result_peaks
#    return True