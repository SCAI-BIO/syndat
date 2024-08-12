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
    long_description_content_type='text/markdown',
    install_requires=[
        'pandas~=2.1.4',
        'numpy~=1.26.2',
        'scipy~=1.11.4',
        'scikit-learn~=1.3.2',
        'matplotlib~=3.8.2',
        'seaborn~=0.13.0',
        'setuptools==69.0.2'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    include_package_data=True,  # Ensure non-Python files are included
    python_requires='>=3.9',  # Specify minimum Python version
    keywords='synthetic-data',
    project_urls={
        'Documentation': 'https://github.com/SCAI-BIO/syndat#readme',
        'Source': 'https://github.com/SCAI-BIO/syndat',
        'Tracker': 'https://github.com/SCAI-BIO/syndat/issues',
    },
)
