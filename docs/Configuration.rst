Configuration
=================

tomatos is (currently) designed to expect flat ntuples of root files like in `tests/files/ <https://github.com/fred-renner/tomatos/tree/master/tests/files>`_ with the structure 

.. code-block:: rst

    ntuple_path/SAMPLE/SOME_SYSTEMATIC.root


The configuration scheme starts with a yaml file (`examples <https://github.com/fred-renner/tomatos/tree/master/configs>`_ ) which is then further preprocessed in `config.py <https://github.com/fred-renner/tomatos/blob/master/tomatos/config.py>`_ which for instance gathers the relevant paths for all systematics. 
