.. _developer_installation:

======================
Developer installation
======================

Requirements
============

The requirements are listed in :ref:`user_install`.

Additionally, the following Python libraries may be required:

- sphinx: documentation generator
- sphinx-rtd-theme: theme for sphinx
- pytest: test runner
- setuptools (conda package): packaging tools
- conda-build (root environment only): conda packaging tools

Installation
============

Follow :ref:`source_install` for a developer installation (use editable inplace installation).

arim can be rebuilt at any time using::

  python setup.py build_ext --inplace

Finally, :ref:`check all tests pass <run_tests>`. If needed, :ref:`build the documentation <build_doc>`.

