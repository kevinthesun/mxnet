#!/bin/bash
#Build mxnet and docs
cp make/config.mk .
make -j8 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 || exit 1
cd python
sudo python setup.py install
cd ../docs
make html

#Setup virtualenv and install packages
virtualenv ENV
source /workspace/ENV/bin/activate

pip install requests
pip install jupyter
pip install graphviz
pip install matplotlib

cd ../python
python setup.py install

#Test tutorials
cd ../tests/nightly
python test_tutorial.py
