from setuptools import setup, find_packages

from rearrangement_algorithm import __version__, __author__, __email__

with open("README.md") as rm:
    long_desc = rm.read()

#with open("requirements.txt") as req:
#    requirements = req.read().splitlines()

setup(
    name = "rearrangement_algorithm",
    version = __version__,
    author = __author__,
    author_email = __email__,
    description = "Implementation of the rearrangement algorithm",
    long_description=long_desc,
    license='GPLv3',
    url='https://gitlab.com/klb2/rearrangement-algorithm',
    project_urls={
        #'Documentation': "https://digcommpy.readthedocs.io/",
        'Source Code': 'https://gitlab.com/klb2/rearrangement-algorithm'
        },
    packages=find_packages(),
    tests_require=['pytest', 'tox'],
    install_requires=['numpy', 'scipy'],
)
