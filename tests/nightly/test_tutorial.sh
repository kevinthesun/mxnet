#!/bin/bash

#Setup virtualenv and install pip package
virtualenv ENV

pip install --upgrade setuptools
pip install numpy
pip install request
pip install jupyter
pip install graphviz
pip install matplotlib

#Test tutorials
python tests/nightly/test_tutorial.py
