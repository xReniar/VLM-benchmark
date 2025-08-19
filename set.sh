#!/bin/bash

sudo apt install gcc g++ zip unzip nano
sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.570.169 /usr/lib/x86_64-linux-gnu/libcuda.so
sudo ldconfig

export PYTHONPATH=$PYTHONPATH:/home/rainer/Code/VLM-benchmark
