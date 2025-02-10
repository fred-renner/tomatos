Plotting
==============

After training this uses ``metrics.h5`` to plot:

* Histogram of the best epoch
* loss function
* bandwidth evolution
* relative uncertainty evolution
* cut evolution
* sharp hist deviation from binned kernel density estimate hists
* mp4 movie of kde evaluated histogram


.. code-block:: bash

    tomatos --config <my_config> --plot


