Syndat: synthetic data quality evaluation
=========================================

.. meta::
   :description: Syndat is a Python library to evaluate and visualize the quality of synthetic patient and clinical trial data using distribution, correlation and discrimination metrics.
   :keywords: synthetic data, synthetic patient data, clinical data, synthetic data evaluation, data fidelity, data quality metrics, Jensen-Shannon distance, discriminator AUC

Syndat is a Python library to **evaluate and visualize the quality of synthetic
patient and clinical trial data**. It computes synthetic data quality metrics for
tabular datasets — Jensen-Shannon distance, normalized correlation difference and
discriminator AUC — and aggregates them into three scores (distribution,
correlation and discrimination) on a 0–100 scale. Visualizations cover feature
distributions, correlation structures, summary statistics and SHAP-based
discrimination plots, including stratified raincloud, GOF and trajectory plots
for clinical subgroups.

Syndat is developed at Fraunhofer SCAI and used in the NFDI4Health consortium.
The evaluation methodology is described in
`Adams et al., iScience 2025 <https://doi.org/10.1016/j.isci.2025.112382>`_.

Start with the :doc:`quickstart <quickstart>` to install Syndat and run a first
evaluation, or read the :doc:`metrics <metrics>` page for an explanation of the
quality metrics and how to interpret the resulting scores.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   metrics
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project links:

   GitHub repository <https://github.com/SCAI-BIO/syndat>
   PyPI package <https://pypi.org/project/syndat>
   Zenodo DOI <https://doi.org/10.5281/zenodo.15791976>
   SYNDAT web dashboard <https://syndat.scai.fraunhofer.de>
   SYNDAT Dashboard (privacy assessment) <https://github.com/nfdi4health/syndat-dashboard>
   iScience publication <https://doi.org/10.1016/j.isci.2025.112382>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
