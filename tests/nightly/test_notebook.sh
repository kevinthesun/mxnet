#!/bin/sh
echo "BUILD make"
cp ./make/config.mk .
echo "USE_CUDA=1" >> ./config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >> ./config.mk
echo "USE_CUDNN=1" >> ./config.mk
echo "USE_PROFILER=1" >> ./config.mk
echo "DEV=1" >> ./config.mk
echo "EXTRA_OPERATORS=example/ssd/operator" >> ./config.mk
make clean
make -j$(nproc) || exit -1

echo "BUILD python2 mxnet"
cd ./python
python setup.py install || exit 1

echo "BUILD python3 mxnet"
python3 setup.py install || exit 1
echo "~/.local"
cd ../tests/nightly

echo "Pull mxnet-notebook"
git clone https://github.com/dmlc/mxnet-notebooks.git

echo "Test Jupyter notebook"
python test_ipynb.py

echo "Test Summary Start"
cat test_summary.txt
echo "Test Summary End"
