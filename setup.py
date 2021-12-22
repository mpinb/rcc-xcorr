from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='rcc-xcorr',  # the name used for pip install
    version='0.0.1',
    description='Perform template matching using fast normalized cross-correlation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    py_modules=['rcc-xcorr'],  # the name used when importing the package
    package_dir={'': 'src'},
    requires= [
        'numpy',
        'scipy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    url="https://github.com/research-center-caesar/rcc-xcorr.git",
)