#!/bin/bash

#Test tutorials
make docs
source /ENV/bin/activate
cd tests/nightly
python test_tutorial.py
