.. tomatos documentation master file, created by
   sphinx-quickstart on Fri Feb  7 12:35:43 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tomatos documentation
=====================

TOMATOS is a fully differentiable HEP-analysis optimization framework. It includes the statistical test and can therefore consider the influence of uncertainties while simultaneously optimizing variable cuts, bin edges or a Neural Network. Based on `neos <https://github.com/gradhep/neos>`_, `relaxed <https://github.com/gradhep/relaxed>`_  and `jax <https://github.com/jax-ml/jax>`_. I strongly recommend to read the `jax introduction <https://jax.readthedocs.io/en/latest/tutorials.html>`_. Be reminded that this still at an early stage.

The Code is currently very self-explanatory, always splitting a main function into subtasks. You can just follow the ``cli.py`` which splits into the chapters of this doc. Before anything always get the environment:
.. code-block:: bash

   source ./tomatos_env/bin/activate


.. toctree::
   :hidden:
   :maxdepth: 1

   Configuration
   Preprocessing
   Training
   Plotting

All steps can be done in one go


.. code-block:: bash

   tomatos --config <my_config> --prep --train --plot
