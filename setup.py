import io
import os

from setuptools import setup, find_packages

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
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    url='https://github.com/SCAI-BIO/syndat',
    license='MIT',
    author='Tim Adams',
    author_email='tim.adams@scai.fraunhofer.de',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'pandas~=2.1',
        'numpy~=1.26',
        'scipy~=1.11',
        'scikit-learn~=1.5',
        'matplotlib~=3.8',
        'seaborn~=0.13',
        'shap~=0.42.0',
        'setuptools>=70.0.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    include_package_data=True,  
    python_requires='>=3.9',  
    keywords='synthetic-data, data-quality, data-visualization',
    project_urls={
        'Documentation': 'https://github.com/SCAI-BIO/syndat#readme',
        'Source': 'https://github.com/SCAI-BIO/syndat',
        'Tracker': 'https://github.com/SCAI-BIO/syndat/issues',
    },
)
