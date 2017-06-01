#!/bin/bash
#Build mxnet and docs
cp make/config.mk .
make -j8 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 || exit 1
cd docs
make html

#Setup virtualenv and install packages
cd ../python
virtualenv ENV
source /workspace/python/ENV/bin/activate
pip install six
/workspace/python/ENV/bin/python setup.py install

pip install requests
pip install jupyter
pip install graphviz
pip install matplotlib

#Test tutorials
cd ../tests/nightly
python test_tutorial.py
