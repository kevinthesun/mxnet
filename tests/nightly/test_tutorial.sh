#!/bin/bash

#Test tutorials
make docs
source ~/ENV/bin/activate
python tests/nightly/test_tutorial.py
