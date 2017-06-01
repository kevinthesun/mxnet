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
cp make/config.mk .
make -j8 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 || exit 1
cd python
python setup.py install
cd ..
make docs
python tests/nightly/test_tutorial.py
