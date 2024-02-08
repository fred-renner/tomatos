WIP! ATOS - AuTomated Optimization on  $CL_s$ Sensitivity for a boosted HH4b VBF analysis. Based on NEOS by Nathan Simpson ([paper](https://arxiv.org/pdf/2203.05570.pdf), [repo](https://github.com/gradhep/neos)). Only works with the exact requirements in the toml. Requires dumped ntuples from [pyhh](https://gitlab.cern.ch/frenner/pyhh).

```bash
git clone https://gitlab.cern.ch/frenner/atos.git
# do the next two lines if you want to work from a virtual environment
python -m venv ./atos_env
source ./atos_env/bin/activate
pip install --upgrade pip
cd atos
# avoid a conflict
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True 
# install dependencies
pip install --editable .

atos
```