auTOMATed Optimization on Sensitivity - tomatos - for a boosted HH4b VBF analysis. Based on NEOS by Nathan Simpson ([paper](https://arxiv.org/pdf/2203.05570.pdf), [repo](https://github.com/gradhep/neos)). Only works with the exact requirements in the toml. Requires dumped ntuples from [pyhh](https://gitlab.cern.ch/frenner/pyhh).

I strongly recommend to read [how to think in jax](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html) and [JAX - The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).

```bash
git clone https://gitlab.cern.ch/frenner/tomatos.git
# do the next two lines if you want to work from a virtual environment
python -m venv ./tomatos_env
source ./tomatos_env/bin/activate
pip install --upgrade pip
cd tomatos
# avoid a conflict
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True 
# install dependencies
pip install --editable .

tomatos
```