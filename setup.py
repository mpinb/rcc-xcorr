from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='rcc-xcorr',  # the name used for pip install
    version='1.0.1',
    description='Perform template matching using fast normalized cross-correlation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    py_modules=['rcc_xcorr'],  # the name used when importing the package
    #packages=find_packages(exclude='tests'),
    packages=['rcc_xcorr',
              'rcc_xcorr.xcorr'],
    #package_dir={'': 'src'},
    requires=[
        #'python=3.8',
        'cupy',
        'numpy',
        'scipy',
        'tqdm',
        'GPUtil',
        # additional libraries used by the test and benchmarking scripts
        'perfplot',
        'tifffile',
        'dill',
        'matplotlib'
    ],
    url="https://github.com/research-center-caesar/rcc-xcorr.git",
    author="Omar Valerio",
    author_email="omar.valerio@mpinb.mpg.de",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)