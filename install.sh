#!/bin/bash

#yay -S cmake boost eigen lpsolve --noconfirm
git clone https://github.com/Svalorzen/AI-Toolbox
cd AI-Toolbox
mkdir build
cd build
cmake cmake -DMAKE_POMDP=1 -DMAKE_PYTHON=1 -DPYTHON_VERSION=3 ..
make
cp AI-Toolbox/build/AI-Toolbox.so .
