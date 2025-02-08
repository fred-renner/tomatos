Preprocessing 
==============

The preprocessing does:
* prepare data into a 3D array of size (n_samples, n_batch_events, n_vars)
* Min-Max scaling (only the``vars`` setup in the config)
* upscale events for training (accounted for by scale factors in histograms) 

It starts by collecting data from all samples, figures their min-max range and creates a main stack in ``data.h5`` containing a 2D array (n_total_events, n_vars) for each sample. In a second step it creates ``train.h5``, ``valid.h5`` and ``test.h5`` according to the configured splitting. It concludes with writing metadata, in particular values that define the min-max scaling.