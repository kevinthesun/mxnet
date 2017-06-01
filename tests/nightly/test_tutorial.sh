#!/bin/bash
Build mxnet and docs
cp make/config.mk .
make -j8 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 || exit 1
cd python
sudo python setup.py install
cd ../docs
make html

#Setup virtualenv and install packages
cd ../tests/nightly
virtualenv ENV
source /workspace/tests/nightly/ENV/bin/activate
pip install six

pip install requests
pip install jupyter
pip install graphviz
pip install matplotlib
sudo python ../../python/setup.py install

#Test tutorials
python test_tutorial.py
