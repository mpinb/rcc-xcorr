# rcc-xcorr
A Python library to compute normalized 2D cross-correlation of images using GPU and multiprocessing.

## Installation
```python
pip install rcc-xcorr
```

## Usage

The rcc-xcorr tool minimal input is a list of correlation, an array of images and an array of templates. 
The correlation list is a list of pairs, the first one correspond to the image_id (the image index within the images array),
the second correspond to a template_id (the template index within the templates array).

Example:

correlations : [[0 1]
                [0 2] 
               [1 3]
                [1 2]
                [0 4]
                [2 0]
]

````
import rcc-xcorr

correlations = [ ....  ]
images =   []
templates = []

rcc-xcorr.perform_correlations(correlations, images, templates)

````


Note: The order in which the correlations appear in the array is the same order in which 
the correlation results are returned to the caller program. (maybe a feature??)

The program assumes all given image_id and template_id are valid indices inside the images/templates array.

## Options

The following options can be used to control the execution of the rcc-xcorr tool

### normalize_input

### crop_output

### use_gpu

The program also supports a CPU-ONLY mode, which can be used in case the parallel system does not
count with a GPU. This also allows to have several instances of the rcc-xcorr tool running on a
given node. One of them using the GPU resources and a second one constrained only to use the CPU cores.


### group_correlations

This optimization can be used, in case the list of correlations contains several instances of correlations where a single image is
compared against a given number of templates.

In those cases the rcc-xcorr tool will first group the correlations having the same image.
This optimization avoids duplicate transfer of data between the CPU and the GPU memory.

NOTE: the final output preserves the order in which the correlations where originally entered.


### num_workers

In case the use_gpu option is set to True the value used for num_workers should match the num_gpus value (default=4)
The workers are used in a round-robin fashion to dispatch individual correlation pairs to the device (GPU) in order
to keep the GPU(s) working at their full capacity a sufficient amount of workers per GPU must be in place.
The amount of workers that would make GPU work at its full capacity depend on the GPU itself.
A rule of thumb is to use 3 workers per GPU device in soma. Increasing the number of workers above
the number for optimum performance will degrade the performance of the device due to context switching from the GPU
scheduler.

### num_gpus

By default the rcc-xcorr tool will use all the devices allocated by the job 
(all the devices available on the node). If the num_gpus option is passed then the tool will
restrict itself to use the indicated number of devices at most. In case less devices
are available then this value is adjusted and the final value is reported to the user.
                    