from setuptools import setup

setup(
    name='scarlet-test-data',
    version='0.1.0',    
    description='Scarlet, test data storage',
    url='https://github.com/astro-data-lab/scarlet-test-data',
    author='Peter Melchior',
    author_email='melchior@astro.princeton.edu',
    license='MIT',
    packages=['scarlet_test_data'],
    package_data={
        "scarlet_test_data": ["data/*.npz", "data/*.pkl", "data/test_resampling/*.npz", "data/test_resampling/*.fits", "tests/*"],
    },
    include_package_data=True,
)