## This is WIP!

pyhh is a python only analysis framework processing ntuples from the [easyJet framework](https://gitlab.cern.ch/easyjet/hh4b-analysis/) as inputs to do a boosted VBF HH->4b analysis. It is capable of doing event selection, dumping hists and selected variables and has some scripts for plotting and fitting. Install with:
```bash
git clone https://gitlab.cern.ch/frenner/pyhh.git
# do the next two lines if you want to work from a virtual environment
python -m venv ./pyhh_env
source ./pyhh_env/bin/activate 
cd pyhh
# need this right now
pip install --upgrade pyAMI_core
pip install --upgrade pyAMI_atlas
# install packages
pip install --editable .

```

`pyhh select` is currently the main purpose and it invokes the [selector module](https://gitlab.cern.ch/frenner/pyhh/-/blob/master/pyhh/selector/). After adjusting some output filepaths in [pyhh/selector/configuration.py](https://gitlab.cern.ch/frenner/pyhh/-/blob/master/pyhh/selector/configuration.py) you could do
```bash
pyhh select --file <easyjet_ntuples_with_large_R_and_small_R_jets.root> --fill --dump
``` 

There are two testfiles with 10 events in [tests/test_files/](https://gitlab.cern.ch/frenner/pyhh/-/blob/master/tests/test_files/)