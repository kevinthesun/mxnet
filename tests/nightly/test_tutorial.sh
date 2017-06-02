#!/bin/bash

wget http://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.deb && \
    dpkg -i scala-2.11.8.deb && rm scala-2.11.8.deb

sudo apt-get update
sudo apt-get -y install git
sudo apt-get -y install python-opencv
sudo apt-get -y install ipython ipython-notebook
sudo apt-get -y install graphviz
sudo apt-get -y install doxygen
sudo apt-get -y install pandoc

sudo python -m pip install -U pip
sudo pip install virtualenv
sudo pip install sphinx==1.5.1 CommonMark==0.5.4 breathe mock==1.0.1 recommonmark pypandoc
sudo pip install --upgrade requests

#Build mxnet and docs
cp make/config.mk .
make -j8 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
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
python test_tutorial.py || exit 1
