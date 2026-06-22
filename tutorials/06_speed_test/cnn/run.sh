#!/bin/bash
# python3 -B train.py
# python3 -B \
#     ../../../src/api-core/main.py \
#     --input="." \
#     --output="." \
#     --backend="tensorflow" \
#     --bit=64
# rm -rf .vscode/ ../../../src/api-core/__pycache__ ../../../src/dump_model/__pycache__
# python3 -B data.py
# echo ""
# echo "THIS IS PYTHON INFERNECE"
# python3 -B test.py
# echo ""
echo "THIS IS C++ INFERNECE"
export OMP_NUM_THREADS=1
g++-15 test.cpp -O3 -march=native -ffast-math -fopenmp \
    -I$CONDA_PREFIX/include \
    -L$CONDA_PREFIX/lib \
    -Wl,-rpath,$CONDA_PREFIX/lib \
    -lhdf5_cpp -lhdf5 \
    -o a.out
./a.out
echo ""