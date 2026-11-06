Synthetic data quality metrics
==============================

Syndat evaluates how closely a synthetic dataset reproduces the statistical
properties of the real data it was generated from. Three base metrics are
computed per dataset pair and aggregated into scores between 0 and 100, where a
higher score means better fidelity.

The metrics operate on tabular data (``pandas.DataFrame``) and are designed for
clinical and patient-level data, but apply to any tabular dataset.

Distribution similarity
-----------------------

Measured with the Jensen-Shannon distance, a symmetric and bounded measure of
similarity between two probability distributions. For every feature, Syndat
estimates the distribution in the real and the synthetic data and computes the
distance between them. The distance is 0 for identical distributions and 1 for
disjoint distributions.

.. code-block:: python

    from syndat.metrics import jensen_shannon_distance

    jsd = jensen_shannon_distance(real, synthetic)
    score = syndat.scores.distribution(real, synthetic)

The score is the average over all features, rescaled as ``(1 - mean JSD) * 100``.
Use it to detect features whose value distribution differs between the datasets,
for example due to mode collapse or clipping in the generator.

Correlation similarity
----------------------

A synthetic dataset can match every marginal distribution and still destroy the
relationships between features. The normalized correlation difference compares
the correlation matrices of both datasets using the chosen method (``spearman``
by default, ``pearson`` and ``kendall`` are also supported).

.. code-block:: python

    from syndat.metrics import normalized_correlation_difference

    difference = normalized_correlation_difference(real, synthetic)
    score = syndat.scores.correlation(real, synthetic)

The difference is 0 for identical correlation matrices and 1 for completely
different ones. The score is ``(1 - difference) * 100``. Values above 1 indicate
inverted correlation structure (mostly negative correlations in the real data,
mostly positive in the synthetic data); this worst case is scored as 0.

Discrimination
--------------

A classifier is trained to distinguish real from synthetic records. If the
synthetic data is a good stand-in for the real data, the classifier should not
perform better than chance, so the ROC AUC is close to 0.5.

.. code-block:: python

    from syndat.metrics import discriminator_auc

    auc = discriminator_auc(real, synthetic, n_folds=5)
    score = syndat.scores.discrimination(real, synthetic)

The discrimination score is based on the cross-validated AUC of a Random Forest
classifier and is normalized to 0-100, with 100 meaning indistinguishable
datasets. SHAP values from the same classifier can be visualized with
``syndat.visualization.plot_shap_discrimination`` to see which features drive
the distinction.

Interpreting the scores
-----------------------

.. list-table::
   :header-rows: 1

   * - Score
     - 0
     - 50
     - 100
   * - Distribution
     - feature distributions are disjoint
     - noticeable differences
     - indistinguishable distributions
   * - Correlation
     - correlation structure is unrelated or inverted
     - partially preserved
     - identical correlation matrices
   * - Discrimination
     - classifier separates the data almost perfectly
     - moderate separability
     - classifier performs at chance level

The scores are diagnostic, not a pass/fail criterion: acceptable values depend on
the downstream task. A synthetic dataset used for model development usually needs
high distribution and correlation scores, while a dataset used for exploratory
analysis may tolerate more divergence. The iScience paper
`On the fidelity versus privacy and utility trade-off of synthetic patient data
<https://doi.org/10.1016/j.isci.2025.112382>`_ discusses these trade-offs on
clinical datasets.

Stratified evaluation
---------------------

For clinical data, aggregate scores can hide subgroup-specific problems. Syndat
provides stratified visualizations (raincloud plots, GOF plots and trajectory
comparisons) to inspect quality across subgroups; see
``examples/rct_example.py`` and the API reference for
:mod:`syndat.rct`.
