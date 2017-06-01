#!/bin/bash

#Setup virtualenv and install packages
virtualenv ENV
source workspace/ENV/bin/activate

pip install --upgrade setuptools
pip install requests
pip install jupyter
pip install graphviz
pip install matplotlib

#Test tutorials
make docs
cd tests/nightly
python test_tutorial.py
