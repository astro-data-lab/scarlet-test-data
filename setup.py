from setuptools import setup

import os
import importlib
import pip

module_name = "git"
package = "gitpython"

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

try:
    import git
except ModuleNotFoundError:
    install(package)
finally:
    globals()[module_name] = importlib.import_module(module_name)

def pull_first():
    """This script is in a git directory that can be pulled."""
    cwd = os.getcwd()
    gitdir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(gitdir)
    g = git.cmd.Git(gitdir)
    try:
        g.execute(['git', 'lfs', 'pull'])
    except git.exc.GitCommandError:
        raise RuntimeError("Make sure git-lfs is installed!")
    os.chdir(cwd)

pull_first()

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