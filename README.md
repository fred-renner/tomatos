# auTOMATed Optimization on Sensitivity
TOMATOS is a fully differentiable HEP-analysis optimization framework. It includes the statistical test and can therefore consider the influence of uncertainties. It supports the simultaneous optimization of variable cuts, bin edges and a Neural Network. 

# Documentation
https://tomatos.readthedocs.io

# Installation
```bash
git clone https://gitlab.cern.ch/frenner/tomatos.git
python3.9 -m venv ./tomatos_env
source ./tomatos_env/bin/activate
pip install --upgrade pip
# avoid a conflict
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True 
# install dependencies
pip install --editable ./tomatos
# install jaxlib from google servers
# cpu
pip install jaxlib==0.3.14 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# gpu
# pip install jaxlib==0.3.14+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

```

# run 
After assigning the two paths at the beginning of the config/demo_*.yaml you can do for example
```bash
source ./tomatos_env/bin/activate
tomatos --config ./tomatos/configs/demo_cls_nn.yaml --prep --train --plot
```