#!/bin/bash

#Test tutorials
make docs
cd tests/nightly
python test_tutorial.py
