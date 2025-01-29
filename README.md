# auTOMATed Optimization on Sensitivity
Based on [neos](https://github.com/gradhep/neos) and [relaxed]. I strongly recommend to read the [jax introduction](https://jax.readthedocs.io/en/latest/tutorials.html) if you want to contribute.

# Installation
dependency installation not ideal currently, will change in the future.
```bash
git clone https://gitlab.cern.ch/frenner/tomatos.git
python3.9 -m venv ./tomatos_env
source ./tomatos_env/bin/activate
pip install --upgrade pip
cd tomatos
# avoid a conflict
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True 
# install dependencies
pip install --editable .

# need to install them by hand from google servers, as they don't maintain older versions on PyPI
# cpu
# pip install jaxlib==0.3.14 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# gpu
# pip install jaxlib==0.3.14+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


tomatos
```


