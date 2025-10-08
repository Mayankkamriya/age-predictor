#!/bin/bash
# build.sh

# Install Python 3.11
pyenv install 3.11.9 -s
pyenv global 3.11.9

# Install dependencies
pip install -r requirements.txt