import io
import os

from setuptools import setup

DESCRIPTION = 'A library for evaluation & visualization of synthetic data.'

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name='syndat',
    version='0.0.3',
    packages=['syndat'],
    url='https://github.com/SCAI-BIO/syndat',
    license='CC BY-NC-ND 4.0.',
    author='Tim Adams',
    author_email='tim.adams@scai.fraunhofer.de',
    description=DESCRIPTION,
    long_description=DESCRIPTION,
)
