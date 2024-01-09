WIP! Auto-optimization on $CL_s$ for a boosted HH4b VBF analysis. Based on NEOS by Nathan Simpson ([paper](https://arxiv.org/pdf/2203.05570.pdf), [repo](https://github.com/gradhep/neos)). Only works with the exact requirements in the toml. Requires dumped ntuples from [pyhh](https://gitlab.cern.ch/frenner/pyhh).

```bash
git clone https://gitlab.cern.ch/frenner/hh_neos.git
# do the next two lines if you want to work from a virtual environment
python -m venv ./neos_env
source ./neos_env/bin/activate
cd hh_neos
# avoid a conflict
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True 
# install dependencies
pip install --editable .

hh_neos
```