from setuptools import setup, find_packages

setup(
    name='beatLab',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0.0',
        'pandas',
        'hdf5storage',
        'numpy==1.26.4',
        'tqdm',
        'matplotlib',
        'scipy',
        'requests',
        'h5py',
        'keras-complex',#' @ git+ssh://git@github.com/JesperDramsch/keras-complex.git',
        'numba'
    ],
    include_package_data=True,
)